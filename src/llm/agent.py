"""
LLM Agent with RAG Tool Integration

Uses Google Gemini free tier as the primary LLM.
Falls back to RAG-only template responses if no LLM key is available.
ChromaDB-backed knowledge base with 16 customer support documents.
"""

import os
import logging
from typing import Optional, Dict, Any, List

from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

# Detect available LLM backends
LLM_BACKEND = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    if os.getenv("GOOGLE_API_KEY"):
        LLM_BACKEND = "google"
except ImportError:
    pass

if not LLM_BACKEND:
    try:
        from langchain_groq import ChatGroq
        if os.getenv("GROQ_API_KEY"):
            LLM_BACKEND = "groq"
    except ImportError:
        pass

if not LLM_BACKEND:
    try:
        from langchain_openai import ChatOpenAI
        if os.getenv("OPENAI_API_KEY"):
            LLM_BACKEND = "openai"
    except ImportError:
        pass


class BaseAgent:
    """Base class for LLM agents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_initialized = False

    async def initialize(self) -> None:
        raise NotImplementedError

    async def process_query(self, text: str, **kwargs) -> str:
        raise NotImplementedError

    async def cleanup(self) -> None:
        raise NotImplementedError


class CustomerSupportAgent(BaseAgent):
    """
    Customer support agent with RAG-powered knowledge retrieval.

    LLM priority:
    1. Google Gemini free tier  (GOOGLE_API_KEY)
    2. OpenAI                   (OPENAI_API_KEY)
    3. RAG-only fallback        (no key needed)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.llm = None
        self.agent = None
        self.agent_executor = None
        self.knowledge_base = None
        self.collection = None
        self.chroma_client = None
        self.embedding_model = None
        self.llm_backend = None

    async def initialize(self) -> None:
        """Initialize LLM, knowledge base, tools, and agent."""
        # Step 1: Initialize LLM
        await self._init_llm()

        # Step 2: Initialize knowledge base (ChromaDB)
        await self._setup_knowledge_base()

        # Step 3: Create tools including RAG tool
        tools = self._create_tools()

        # Step 4: Create agent (only if LLM is available)
        if self.llm:
            self._create_agent(tools)

        self.is_initialized = True
        logger.info(
            f"Agent initialized (LLM: {self.llm_backend or 'RAG-only'})"
        )

    async def _init_llm(self) -> None:
        """Detect and initialize the best available free LLM."""
        google_key = os.getenv("GOOGLE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        temperature = self.config.get("temperature", 0.7)

        if groq_key:
            try:
                from langchain_groq import ChatGroq
                self.llm = ChatGroq(
                    model_name=self.config.get("model", "llama-3.3-70b-versatile"),
                    groq_api_key=groq_key,
                    temperature=temperature,
                )
                self.llm_backend = "groq"
                logger.info("LLM: Groq (Llama 3)")
                return
            except Exception as e:
                logger.warning(f"Groq init failed: {e}")

        if google_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(
                    model=self.config.get("model", "gemini-2.0-flash"),
                    google_api_key=google_key,
                    temperature=temperature,
                )
                self.llm_backend = "google-gemini"
                logger.info("LLM: Google Gemini (free tier)")
                return
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}")

        if openai_key:
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model=self.config.get("model", "gpt-3.5-turbo"),
                    openai_api_key=openai_key,
                    temperature=temperature,
                )
                self.llm_backend = "openai"
                logger.info("LLM: OpenAI")
                return
            except Exception as e:
                logger.warning(f"OpenAI init failed: {e}")

        # No LLM key — system will use RAG-only fallback
        logger.warning(
            "No LLM API key found. Running in RAG-only mode. "
            "Set GOOGLE_API_KEY (free) or OPENAI_API_KEY in .env"
        )
        self.llm = None
        self.llm_backend = None

    # ------------------------------------------------------------------ #
    #  Knowledge Base Setup (ChromaDB)                                     #
    # ------------------------------------------------------------------ #

    async def _setup_knowledge_base(self) -> None:
        """Set up ChromaDB with the 16 customer-support documents."""
        try:
            import chromadb
            import hashlib

            db_path = "./data/chroma_db"
            os.makedirs(db_path, exist_ok=True)

            self.chroma_client = chromadb.PersistentClient(path=db_path)
            collection_name = "customer_support_kb"

            # Re-use existing collection if it has data
            try:
                self.collection = self.chroma_client.get_collection(
                    collection_name
                )
                if self.collection.count() > 0:
                    logger.info(
                        f"KB loaded: {self.collection.count()} documents"
                    )
                    return
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={
                        "description": "Customer support knowledge base"
                    },
                )

            # Ingest predefined documents
            docs = self._get_customer_support_documents()
            documents, metadatas, ids = [], [], []

            for i, doc in enumerate(docs):
                doc_id = (
                    f"doc_{i}_"
                    f"{hashlib.md5(doc['content'].encode()).hexdigest()[:8]}"
                )
                documents.append(doc["content"])
                metadatas.append(
                    {
                        "category": doc["category"],
                        "title": doc["title"],
                        "doc_id": doc_id,
                    }
                )
                ids.append(doc_id)

            self.collection.add(
                documents=documents, metadatas=metadatas, ids=ids
            )
            logger.info(f"Ingested {len(documents)} documents into ChromaDB")

        except Exception as e:
            logger.error(f"Knowledge base setup failed: {e}")
            raise

    # ------------------------------------------------------------------ #
    #  RAG Search — queries ChromaDB for relevant documents                #
    # ------------------------------------------------------------------ #

    def _rag_search(self, query: str) -> str:
        """
        Retrieve top-3 relevant documents from ChromaDB.

        Returns a formatted string with category, title, and content.
        Filters out low-similarity results (distance > 1.5).
        """
        if not hasattr(self, "collection") or self.collection is None:
            return (
                "Knowledge base not available. "
                "Please ensure the service is properly initialized."
            )

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=3,
                include=["documents", "metadatas", "distances"],
            )

            if (
                not results["documents"]
                or not results["documents"][0]
            ):
                return ""

            formatted_results = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # Filter low-relevance results
                if dist > 1.5:
                    continue

                formatted_results.append(
                    f"[{meta['category'].upper()}] {meta['title']}\n{doc}"
                )

            if not formatted_results:
                return (
                    "No sufficiently relevant information found "
                    "for your query."
                )

            return "\n\n---\n\n".join(formatted_results)

        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return f"Error searching knowledge base: {e}"

    # ------------------------------------------------------------------ #
    #  Tool & Agent Construction                                           #
    # ------------------------------------------------------------------ #

    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools including the RAG tool."""
        rag_tool = Tool(
            name="knowledge_search",
            description=(
                "Search the customer support knowledge base for "
                "information about returns, shipping, support, "
                "payments, warranties, orders, and products."
            ),
            func=self._rag_search,
        )
        return [rag_tool]

    def _create_agent(self, tools: List[Tool]) -> None:
        """Build a ReAct agent with the given tools."""
        if not self.llm:
            return

        try:
            from langchain.agents import create_react_agent, AgentExecutor

            prompt_template = """You are a helpful and professional customer support agent.
Use the available tools to find accurate information before answering.
Always be polite, concise, and helpful.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

            prompt = PromptTemplate.from_template(prompt_template)

            self.agent = create_react_agent(self.llm, tools, prompt)
            self.agent_executor = AgentExecutor.from_agent_and_tools(
                agent=self.agent,
                tools=tools,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=3,
                max_execution_time=30,
            )
            logger.info("ReAct agent created successfully")

        except Exception as e:
            logger.error(f"Agent creation failed: {e}")
            self.agent = None
            self.agent_executor = None

    # ------------------------------------------------------------------ #
    #  Query Processing                                                    #
    # ------------------------------------------------------------------ #

    async def process_query(self, text: str, **kwargs) -> str:
        """
        Process a user query.

        If an LLM agent is available, run the full ReAct loop.
        Otherwise, return RAG results with a template wrapper.
        """
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized")

        if not text or not text.strip():
            return "I didn't receive a question. Could you please try again?"

        language = kwargs.get("language", "en").lower()

        # Path 1: Full agent with LLM
        if self.agent_executor:
            try:
                agent_input = text
                if language == "hi":
                    agent_input = (
                        "IMPORTANT: You MUST respond ONLY in HINDI (हिन्दी), "
                        "using Devanagari script, even if the customer asks in English.\n\n"
                        f"Customer Question: {text}"
                    )
                result = await self.agent_executor.ainvoke(
                    {"input": agent_input}
                )
                response = result.get("output", "")
                if response:
                    return response
            except Exception as e:
                logger.warning(f"Agent execution failed: {e}")
                # Fall through to RAG-only path

        # Path 2: Direct LLM call without agent (simpler, fewer failure modes)
        if self.llm:
            try:
                rag_context = self._rag_search(text)
                
                # Check if we have actual content in RAG context
                has_rag = "No relevant information" not in rag_context and "Error" not in rag_context
                
                lang_instruction = ""
                if language == "hi":
                    lang_instruction = "IMPORTANT: You MUST respond ONLY in HINDI (हिन्दी), even if the customer asks the question in English."
                
                system_prompt = (
                    "You are a professional and friendly Customer Support Assistant. "
                    f"{lang_instruction}\n\n"
                    "Below is the official knowledge base information for our company.\n\n"
                    f"KNOWLEDGE BASE:\n{rag_context if has_rag else 'No specific policy found for this query.'}\n\n"
                    "INSTRUCTIONS:\n"
                    "1. If the knowledge base contains the answer, use it to provide an accurate response.\n"
                    "2. If the user is just saying hello or making general conversation, respond politely as a support agent.\n"
                    "3. If the user asks about a company policy NOT in the knowledge base, provide a helpful general answer and advise them to contact support at 1-800-HELP-NOW.\n"
                    "4. Keep your response professional, concise, and helpful."
                )
                
                prompt = f"{system_prompt}\n\nCustomer Question: {text}\n\nAssistant Response:"
                response = await self.llm.ainvoke(prompt)
                return response.content
            except Exception as e:
                logger.warning(f"Direct LLM call failed: {e}")

        # Path 3: RAG-only fallback (no LLM at all)
        rag_results = self._rag_search(text)
        if "No relevant" in rag_results or "Error" in rag_results:
            return (
                "I'm sorry, I couldn't find relevant information "
                "for your query. Please contact our support team "
                "at 1-800-HELP-NOW (1-800-435-7669) or email "
                "support@company.com for further assistance."
            )

        return (
            f"Based on our knowledge base, here's what I found:\n\n"
            f"{rag_results}\n\n"
            f"If you need further assistance, please contact our "
            f"support team at 1-800-HELP-NOW."
        )

    async def cleanup(self) -> None:
        """Release all agent resources."""
        self.llm = None
        self.agent = None
        self.agent_executor = None
        self.collection = None
        self.chroma_client = None
        self.is_initialized = False
        logger.info("Agent cleaned up")

    # ------------------------------------------------------------------ #
    #  Knowledge Base Documents                                            #
    # ------------------------------------------------------------------ #

    def _get_customer_support_documents(self) -> List[Dict[str, str]]:
        """16 predefined customer support knowledge base documents."""
        return [
            {
                "title": "Return Policy Overview",
                "category": "returns",
                "content": "We offer a 30-day return policy for all products purchased from our store. Items must be in original condition with all tags and packaging intact. Returns are processed within 5-7 business days of receiving the returned item. Refunds are issued to the original payment method.",
            },
            {
                "title": "Return Process Steps",
                "category": "returns",
                "content": "To initiate a return: 1) Log into your account and go to Order History, 2) Select the order and click 'Return Items', 3) Choose the items to return and reason, 4) Print the prepaid return label, 5) Pack items securely and attach the label, 6) Drop off at any UPS location or schedule pickup.",
            },
            {
                "title": "Non-Returnable Items",
                "category": "returns",
                "content": "The following items cannot be returned: personalized or customized products, perishable goods, digital downloads, gift cards, intimate apparel, and items marked as final sale. Health and safety regulations prevent returns of opened cosmetics and personal care items.",
            },
            {
                "title": "Shipping Methods and Times",
                "category": "shipping",
                "content": "We offer multiple shipping options: Standard shipping (5-7 business days, free on orders over $50), Express shipping (2-3 business days, $12.99), Next-day shipping (1 business day, $24.99). All orders placed before 2 PM EST ship the same day.",
            },
            {
                "title": "International Shipping",
                "category": "shipping",
                "content": "We ship internationally to over 50 countries. International shipping takes 7-14 business days via DHL Express. Shipping costs vary by destination and are calculated at checkout. Customers are responsible for customs fees and import duties. Some restrictions apply to certain products and countries.",
            },
            {
                "title": "Order Tracking",
                "category": "shipping",
                "content": "Once your order ships, you'll receive a tracking number via email. Track your package using the tracking number on our website or the carrier's website. You can also track orders by logging into your account and viewing Order History. Tracking updates may take 24 hours to appear.",
            },
            {
                "title": "Contact Information",
                "category": "support",
                "content": "Customer support is available 24/7 via multiple channels: Phone: 1-800-HELP-NOW (1-800-435-7669), Email: support@company.com, Live chat on our website (available 6 AM - 12 AM EST), or submit a support ticket through your account dashboard.",
            },
            {
                "title": "Response Times",
                "category": "support",
                "content": "Our support team response times: Live chat - immediate during business hours, Phone support - average wait time under 3 minutes, Email support - response within 4 hours during business days, Support tickets - response within 24 hours. Premium customers receive priority support with faster response times.",
            },
            {
                "title": "Product Warranty",
                "category": "warranty",
                "content": "All products come with a manufacturer's warranty. Electronics have 1-year warranty covering defects and malfunctions. Apparel and accessories have 90-day warranty against material defects. Warranty claims require proof of purchase and must be initiated within the warranty period.",
            },
            {
                "title": "Technical Support",
                "category": "technical",
                "content": "Free technical support is available for all electronic products. Our certified technicians provide assistance with setup, troubleshooting, and software issues. Technical support is available Monday-Friday 8 AM - 8 PM EST via phone or email. We also offer remote assistance for compatible devices.",
            },
            {
                "title": "Account Management",
                "category": "account",
                "content": "Manage your account online: Update personal information and addresses, view order history and tracking, manage payment methods, set communication preferences, download invoices and receipts. Account changes may take up to 24 hours to reflect across all systems.",
            },
            {
                "title": "Order Modifications",
                "category": "orders",
                "content": "Orders can be modified or canceled within 1 hour of placement if not yet processed. Contact customer service immediately to make changes. Once an order is processed and shipped, it cannot be modified. You can return unwanted items following our return policy.",
            },
            {
                "title": "Payment Methods",
                "category": "payment",
                "content": "We accept all major credit cards (Visa, MasterCard, American Express, Discover), PayPal, Apple Pay, Google Pay, and Buy Now Pay Later options through Klarna and Afterpay. Gift cards and store credit can also be used for purchases. Payment is processed securely using 256-bit SSL encryption.",
            },
            {
                "title": "Billing and Invoices",
                "category": "billing",
                "content": "Billing occurs when your order ships. You'll receive an email confirmation with invoice details. Invoices are available in your account under Order History. For business purchases, we can provide detailed invoices with tax information. Contact our billing department for any payment disputes or questions.",
            },
            {
                "title": "Product Availability",
                "category": "products",
                "content": "Product availability is updated in real-time on our website. If an item shows as 'In Stock', it's available for immediate shipping. 'Limited Stock' means fewer than 10 items remaining. 'Pre-order' items will ship on the specified release date. Out of stock items can be added to your wishlist for restock notifications.",
            },
            {
                "title": "Size and Fit Guide",
                "category": "products",
                "content": "Each product page includes detailed size charts and fit information. For apparel, we recommend checking measurements against our size guide rather than relying on size labels from other brands. If you're between sizes, we generally recommend sizing up. Our customer service team can provide personalized fit recommendations.",
            },
        ]