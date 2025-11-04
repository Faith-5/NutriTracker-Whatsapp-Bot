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

    # ✅ LangChain Requires This Name
    # Required by LangChain
    def get_messages(self):
        return self.messages

    # Required by LangChain
    def add_messages(self, messages):
        for message in messages:
            self.short.add_message(message)

            if isinstance(message, HumanMessage):
                self.last_messages.append(message.content)
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

            # save summary every X messages
            self.write_counter += 1
            if self.write_counter >= 4:
                last_user = next((m for m in reversed(self.short.messages) if isinstance(m, HumanMessage)), None)
                last_bot = next((m for m in reversed(self.short.messages) if isinstance(m, AIMessage)), None)
                if last_user and last_bot:
                    self.long.save_context({"input": last_user.content}, {"output": last_bot.content})
                self.write_counter = 0


        # Move to long-term memory periodically
        self.write_counter += 1
        if self.write_counter >= 4:
            last_user = next((m for m in reversed(self.short.messages) if isinstance(m, HumanMessage)), None)
            last_bot = next((m for m in reversed(self.short.messages) if isinstance(m, AIMessage)), None)

            if last_user and last_bot:
                self.long.save_context(
                    {"input": last_user.content},
                    {"output": last_bot.content}
                )

            self.write_counter = 0

    # ✅ Safe repetition similarity check
    def _detect_repetition(self):
        if len(set(self.last_messages)) == 1:
            return True

        similar = 0
        for i in range(len(self.last_messages) - 1):
            msg1 = set(self.last_messages[i].lower().split())
            msg2 = set(self.last_messages[i+1].lower().split())
            union = msg1 | msg2

            if len(union) == 0:
                continue

            if len(msg1 & msg2) / len(union) > 0.7:
                similar += 1

        return similar >= 2

    def _reset_context(self):
        last_user = next((m for m in reversed(self.short.messages) if isinstance(m, HumanMessage)), None)
        last_bot = next((m for m in reversed(self.short.messages) if isinstance(m, AIMessage)), None)

        self.clear()

        # Inform user
        self.short.add_message(AIMessage(content="It looks like we're repeating ourselves. Let's restart."))

        # Restore last turn for continuity
        if last_user: self.short.add_message(last_user)
        if last_bot: self.short.add_message(last_bot)

    def clear(self):
        self.short = ChatMessageHistory()
        self.long = ConversationSummaryBufferMemory(
            llm=llm, max_token_limit=200, return_messages=True
        )
        self.last_messages = []
        self.repetition_count = 0
        self.write_counter = 0


# ---- SESSION STORE (use phone number as ID in WhatsApp) ----
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


# ---- LLM CHAIN ----
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


# ---- BOT ENTRY FUNC ----
def run_bot(message: str, session_id: str):
    response = chain_with_history.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content if hasattr(response, "content") else str(response)