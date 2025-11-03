from .prompt import SYSTEM_PROMPT
from app.config import GOOGLE_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_core.messages import HumanMessage, AIMessage


# ---- MODEL ----
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=GOOGLE_API_KEY,
    temperature=0.4,
    max_output_tokens=512
)


# ---- HYBRID MEMORY ----
class HybridMemory:
    def __init__(self):
        self.short = ChatMessageHistory()
        self.long = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=200,
            return_messages=True
        )
        self.write_counter = 0
        self.last_messages = []
        self.repetition_count = 0

    @property
    def messages(self):
        long_hist = self.long.load_memory_variables({})["history"]
        short_hist = self.short.messages
        return long_hist + short_hist

    # ✅ REQUIRED for RunnableWithMessageHistory
    def get_messages(self):
        return self.messages

    def add_messages(self, messages):
        for msg in messages:
            self.short.add_message(msg)

            if isinstance(msg, HumanMessage):
                self.last_messages.append(msg.content)
                if len(self.last_messages) > 3:
                    self.last_messages.pop(0)

                if len(self.last_messages) == 3:
                    if self._detect_repetition():
                        self.repetition_count += 1
                        if self.repetition_count >= 2:
                            self._reset_context()
                            return
                    else:
                        self.repetition_count = 0

        self.write_counter += 1
        if self.write_counter >= 4:
            if len(self.short.messages) >= 2:
                last_user = next((m for m in reversed(self.short.messages) if isinstance(m, HumanMessage)), None)
                last_bot = next((m for m in reversed(self.short.messages) if isinstance(m, AIMessage)), None)
                if last_user and last_bot:
                    self.long.save_context({"input": last_user.content}, {"output": last_bot.content})
            self.write_counter = 0

    def _detect_repetition(self):
        if len(set(self.last_messages)) == 1:
            return True
        similar_count = 0
        for i in range(len(self.last_messages) - 1):
            msg1 = self.last_messages[i].lower()
            msg2 = self.last_messages[i+1].lower()
            if len(set(msg1.split()) & set(msg2.split())) / len(set(msg1.split() + msg2.split())) > 0.7:
                similar_count += 1
        return similar_count >= 2

    def _reset_context(self):
        last_exchange = None
        if len(self.short.messages) >= 2:
            last_user = next((m for m in reversed(self.short.messages) if isinstance(m, HumanMessage)), None)
            last_bot = next((m for m in reversed(self.short.messages) if isinstance(m, AIMessage)), None)
            if last_user and last_bot:
                last_exchange = (last_user, last_bot)

        self.clear()

        self.short.add_message(AIMessage(content="I notice we might be going in circles. Let's start fresh with your question."))

        if last_exchange:
            self.short.add_message(last_exchange[0])
            self.short.add_message(last_exchange[1])

    def clear(self):
        self.short = ChatMessageHistory()
        self.long = ConversationSummaryBufferMemory(
            llm=llm, max_token_limit=200, return_messages=True
        )
        self.write_counter = 0
        self.last_messages = []
        self.repetition_count = 0


session_store = {}
def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = HybridMemory()
    return session_store[session_id]


# ---- PROMPT ----
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=SYSTEM_PROMPT + """

{history}

User: {input}
Assistant:""",
)


# ---- CHAIN ----
chain = (
    {
        "input": RunnablePassthrough(),
        "history": RunnablePassthrough(),
    }
    | prompt
    | llm
)


chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


# ---- BOT FUNCTION ----
def run_bot(message: str, session_id: str):
    response = chain_with_history.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content if hasattr(response, "content") else str(response)
