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

# ---- LIGHTWEIGHT MEMORY ----
class SimpleMemory:
    def __init__(self):
        self.short = ChatMessageHistory()
        self.summary = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=150,  # very small
            return_messages=True
        )

    @property
    def messages(self):
        long_hist = self.summary.load_memory_variables({})["history"]
        short_hist = self.short.messages
        return long_hist + short_hist

    def get_messages(self):
        return self.messages

    def add_messages(self, messages):
        for message in messages:
            self.short.add_message(message)

            # keep last 3 turns only
            if len(self.short.messages) > 6:  # 3 user + 3 bot
                # summarize and flush old short history
                last_user = next((m for m in reversed(self.short.messages) if isinstance(m, HumanMessage)), None)
                last_bot = next((m for m in reversed(self.short.messages) if isinstance(m, AIMessage)), None)

                if last_user and last_bot:
                    self.summary.save_context(
                        {"input": last_user.content},
                        {"output": last_bot.content}
                    )

                # Keep only latest 3 messages
                self.short.messages = self.short.messages[-6:]

# ---- SESSION STORE ----
session_store = {}
def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = SimpleMemory()
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

def run_bot(message: str, session_id: str):
    response = chain_with_history.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content if hasattr(response, "content") else str(response)
