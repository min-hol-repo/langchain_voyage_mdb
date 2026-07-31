"""
MongoDB Troubleshooting RAG (Retrieval-Augmented Generation) System
===================================================================
Tech Stack:
  - LangChain    : RAG pipeline construction
  - Voyage AI    : voyage-4 text embeddings (1024 dimensions)
  - MongoDB Atlas: Document storage + Full-Text Search (works with Free Tier M0)
  - FAISS        : Local vector search (replaces $vectorSearch, no M10+ required)
  - OpenAI       : GPT-4o-mini answer generation
  - Hybrid Search: FAISS vector search + MongoDB $search full-text search
  - RRF          : Reciprocal Rank Fusion result merging

Architecture:
  Query → [FAISS Vector Search]        → Ranked List A  ─┐
        → [MongoDB Atlas $search]      → Ranked List B  ─┤→ RRF → GPT-4o-mini
                                                           └→ Final Answer

Run:
  python mongodb_rag.py
"""

import os
import time
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
import pymongo
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

from langchain_voyageai import VoyageAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# Load environment variables
# ============================================================
load_dotenv()

MONGODB_URI    = os.getenv("MONGODB_URI", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")

# MongoDB settings
DB_NAME           = "mongodb_troubleshooting"
COLLECTION_NAME   = "knowledge_base"
SEARCH_INDEX_NAME = "search_index"   # Atlas Search index (full-text, works on M0)
TEXT_FIELD        = "text"
EMBEDDING_FIELD   = "embedding"

# Model settings
EMBEDDING_MODEL = "voyage-4"
EMBEDDING_DIMS  = 1024
LLM_MODEL       = "gpt-4o-mini"


# ============================================================
# MongoDB Troubleshooting Knowledge Base (Sample Data)
# ============================================================
MONGODB_KNOWLEDGE_BASE = [
    {
        "title": "Connection Pool Exhaustion",
        "category": "connection",
        "source": "MongoDB Connection Management Guide",
        "text": (
            "MongoDB Connection Pool Exhaustion Troubleshooting Guide\n\n"
            "Symptoms:\n"
            "- 'connection pool exhausted' or 'too many connections' errors\n"
            "- Application response delays and timeouts\n"
            "- connections count approaching maxIncomingConnections in mongostat\n\n"
            "Causes:\n"
            "- Code that does not release connections (connection leak)\n"
            "- Sudden traffic spike\n"
            "- maxPoolSize set too low\n"
            "- Slow queries holding connections for too long\n\n"
            "Solutions:\n"
            "1. Check current connection status\n"
            "   db.serverStatus().connections\n\n"
            "2. Adjust maxPoolSize (driver settings)\n"
            "   MongoClient(uri, maxPoolSize=100, waitQueueTimeoutMS=5000)\n\n"
            "3. Detect connection leaks\n"
            "   Use db.currentOp() to find long-running sessions\n\n"
            "Prevention:\n"
            "- Monitor connection pool (Atlas Metrics > Connections)\n"
            "- Set alerts when connections exceed threshold\n"
            "- Optimize slow queries to reduce connection hold time"
        ),
    },
    {
        "title": "Slow Queries due to Missing Indexes",
        "category": "performance",
        "source": "MongoDB Performance Optimization Guide",
        "text": (
            "Slow Queries due to Missing Indexes Troubleshooting Guide\n\n"
            "Symptoms:\n"
            "- Query response time of several seconds or more\n"
            "- COLLSCAN (full collection scan) shown in Atlas Profiler\n"
            "- CPU usage spike\n\n"
            "Diagnosis:\n"
            "1. Check slow query logs\n"
            "   db.system.profile.find({millis: {$gt: 100}}).sort({ts: -1}).limit(10)\n\n"
            "2. Analyze execution plan with explain()\n"
            "   db.collection.find(query).explain('executionStats')\n\n"
            "Solutions:\n"
            "1. Create required indexes\n"
            "   db.orders.createIndex({user_id: 1, created_at: -1})\n\n"
            "2. Compound index design principles (ESR Rule)\n"
            "   - Equality fields first\n"
            "   - Sort fields next\n"
            "   - Range fields last\n\n"
            "Prevention:\n"
            "- Plan indexes when introducing new query patterns\n"
            "- Monitor Atlas Performance Advisor recommendations"
        ),
    },
    {
        "title": "Replication Lag",
        "category": "replication",
        "source": "MongoDB Replication Operations Guide",
        "text": (
            "MongoDB Replication Lag Troubleshooting Guide\n\n"
            "Symptoms:\n"
            "- Large lag shown in rs.printSlaveReplicationInfo()\n"
            "- Stale data returned when reading from Secondary\n\n"
            "Causes:\n"
            "- Excessive write load on Primary\n"
            "- Insufficient resources on Secondary (CPU, I/O)\n"
            "- Limited network bandwidth\n"
            "- Oplog size too small causing rollbacks\n\n"
            "Diagnosis:\n"
            "1. Check replication status\n"
            "   rs.status()\n"
            "   rs.printSlaveReplicationInfo()\n\n"
            "Solutions:\n"
            "1. Immediate action - load distribution\n"
            "   - Temporarily redirect reads to Primary\n"
            "   - Pause batch jobs\n\n"
            "2. Increase Oplog size\n"
            "   mongod.conf: replication.oplogSizeMB: 51200\n\n"
            "Prevention:\n"
            "- Set Oplog size to cover at least 24-72 hours\n"
            "- Keep Secondary server specs equal to Primary"
        ),
    },
    {
        "title": "WiredTiger Cache Memory Shortage",
        "category": "memory",
        "source": "MongoDB Memory Management Guide",
        "text": (
            "WiredTiger Cache Memory Shortage Troubleshooting Guide\n\n"
            "Symptoms:\n"
            "- Sudden query performance degradation\n"
            "- Disk I/O spike\n"
            "- 'WiredTiger eviction' warning messages\n"
            "- Increased server swap usage\n\n"
            "Causes:\n"
            "- Dataset exceeds WiredTiger cache size\n"
            "- Memory settings too low for server specs\n\n"
            "Solutions:\n"
            "1. Increase WiredTiger cache size\n"
            "   mongod.conf:\n"
            "     storage.wiredTiger.engineConfig.cacheSizeGB: 4  # Recommended: 50% of RAM\n\n"
            "2. Reduce memory usage by removing unnecessary indexes\n\n"
            "Prevention:\n"
            "- Monitor cache hit ratio (target: 95%+)\n"
            "- Choose appropriate Atlas instance tier"
        ),
    },
    {
        "title": "Disk Space Shortage",
        "category": "storage",
        "source": "MongoDB Storage Management Guide",
        "text": (
            "MongoDB Disk Space Shortage Troubleshooting Guide\n\n"
            "Symptoms:\n"
            "- 'no space left on device' error\n"
            "- Write operations failing\n"
            "- MongoDB process abnormal termination\n\n"
            "Immediate Actions:\n"
            "1. Check current disk usage\n"
            "   df -h  (OS level)\n"
            "   db.stats()  (MongoDB level)\n\n"
            "Solutions:\n"
            "1. Archive or delete old data\n"
            "   db.logs.deleteMany({createdAt: {$lt: new Date('2024-01-01')}})\n\n"
            "2. Reclaim space with compact command\n"
            "   db.runCommand({compact: 'collection_name'})\n\n"
            "3. Set TTL index for automatic data expiration\n"
            "   db.logs.createIndex({createdAt: 1}, {expireAfterSeconds: 2592000})\n\n"
            "Prevention:\n"
            "- Set alerts when disk usage reaches 80%\n"
            "- Use Atlas Online Archive"
        ),
    },
    {
        "title": "Lock Contention",
        "category": "locking",
        "source": "MongoDB Lock Management Guide",
        "text": (
            "MongoDB Lock Contention Troubleshooting Guide\n\n"
            "Symptoms:\n"
            "- Overall performance degradation\n"
            "- globalLock.ratio value above 0.5\n"
            "- Increased query wait times\n\n"
            "Diagnosis:\n"
            "1. Check currently running operations\n"
            "   db.currentOp({active: true, waitingForLock: true})\n\n"
            "Solutions:\n"
            "1. Kill long-running operations\n"
            "   db.killOp(opid)\n\n"
            "2. Split large operations into smaller batches\n"
            "   - Process 1000 documents at a time with sleep between batches\n\n"
            "Prevention:\n"
            "- Apply batch processing pattern for large operations\n"
            "- Ensure all queries have appropriate indexes"
        ),
    },
    {
        "title": "Primary Election Failure",
        "category": "replication",
        "source": "MongoDB Replica Set Operations Guide",
        "text": (
            "MongoDB Replica Set Primary Election Failure Troubleshooting Guide\n\n"
            "Symptoms:\n"
            "- All members in SECONDARY or UNKNOWN state\n"
            "- Write operations unavailable ('not master' error)\n"
            "- No primary shown in rs.status()\n\n"
            "Causes:\n"
            "- Network partition (split-brain scenario)\n"
            "- Cannot reach majority of voting members\n\n"
            "Solutions:\n"
            "1. Wait for automatic election after network recovery (10-30 seconds)\n\n"
            "2. Adjust member priority to guide Primary election\n"
            "   cfg = rs.conf()\n"
            "   cfg.members[0].priority = 2\n"
            "   rs.reconfig(cfg)\n\n"
            "Prevention:\n"
            "- Always maintain odd number of voting members (3, 5, 7)\n"
            "- Deploy members across different Availability Zones"
        ),
    },
    {
        "title": "Atlas Search Index Error",
        "category": "search",
        "source": "MongoDB Atlas Search Guide",
        "text": (
            "MongoDB Atlas Search Index Error Troubleshooting Guide\n\n"
            "Symptoms:\n"
            "- '$search is not allowed' error\n"
            "- Empty array returned from search\n"
            "- Index status shows FAILED or BUILDING\n\n"
            "Causes:\n"
            "- Atlas Search index not created\n"
            "- Index name mismatch in code\n"
            "- Index still building (status not READY)\n\n"
            "Solutions:\n"
            "1. Verify Atlas Search index status in Atlas UI\n"
            "   Atlas > Database > Search > Index Status (must be READY)\n\n"
            "2. Verify index name matches in code\n"
            "   $search: {index: 'search_index'}  // must match exactly\n\n"
            "3. Wait for index build to complete\n"
            "   Large collections may take several minutes\n\n"
            "Prevention:\n"
            "- Manage index names as code constants\n"
            "- Automate index status checks before deployment"
        ),
    },
    {
        "title": "OOM (Out of Memory) Killer",
        "category": "memory",
        "source": "MongoDB Memory Management Guide",
        "text": (
            "MongoDB OOM (Out of Memory) Killer Troubleshooting Guide\n\n"
            "Symptoms:\n"
            "- MongoDB process suddenly terminates\n"
            "- 'Killed process (mongod)' in system logs\n"
            "- System memory usage reaches 100%\n\n"
            "Causes:\n"
            "- WiredTiger cache + indexes exceed available memory\n"
            "- Memory competition with other processes\n\n"
            "Solutions:\n"
            "1. Limit WiredTiger cache size\n"
            "   cacheSizeGB: 2  (50% or less of available memory)\n\n"
            "2. Configure Swap space\n"
            "   fallocate -l 4G /swapfile && mkswap /swapfile && swapon /swapfile\n\n"
            "Prevention:\n"
            "- Set alerts when memory usage reaches 85%\n"
            "- Choose the appropriate Atlas tier"
        ),
    },
    {
        "title": "Mongodump / Mongorestore Backup and Recovery",
        "category": "backup",
        "source": "MongoDB Backup and Recovery Guide",
        "text": (
            "MongoDB Backup and Recovery Complete Guide\n\n"
            "Mongodump (Backup):\n"
            "1. Full database backup\n"
            "   mongodump --uri='mongodb+srv://...' --out=/backup/$(date +%Y%m%d)\n\n"
            "2. Specific collection backup\n"
            "   mongodump --uri='...' --db=mydb --collection=users --out=/backup/\n\n"
            "Mongorestore (Recovery):\n"
            "1. Full restore\n"
            "   mongorestore --uri='mongodb+srv://...' /backup/20240101/\n\n"
            "2. Faster restore with parallel collections\n"
            "   mongorestore --uri='...' --numParallelCollections=4 /backup/\n\n"
            "Atlas Backup:\n"
            "- Atlas > Cluster > Backup > Take Snapshot\n"
            "- Use Point-in-Time Recovery (PITR)\n"
            "- Continuous Cloud Backup recommended for M10+\n\n"
            "Notes:\n"
            "- Use --oplog option in Replica Set for consistency\n"
            "- Use --gzip option to compress large backups"
        ),
    },
]


# ============================================================
# Component Initialization
# ============================================================

def get_mongodb_client() -> MongoClient:
    """Connect to MongoDB Atlas."""
    if not MONGODB_URI:
        raise ValueError("MONGODB_URI environment variable is not set.")
    client = MongoClient(MONGODB_URI)
    client.admin.command("ping")
    print("✅ Connected to MongoDB Atlas")
    return client


def get_embeddings() -> VoyageAIEmbeddings:
    """Initialize Voyage AI embedding model."""
    if not VOYAGE_API_KEY:
        raise ValueError("VOYAGE_API_KEY environment variable is not set.")
    emb = VoyageAIEmbeddings(voyage_api_key=VOYAGE_API_KEY, model=EMBEDDING_MODEL)
    print(f"✅ Voyage AI embeddings initialized (model: {EMBEDDING_MODEL}, {EMBEDDING_DIMS} dims)")
    return emb


def get_llm() -> ChatOpenAI:
    """Initialize OpenAI LLM."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
    print(f"✅ OpenAI LLM initialized (model: {LLM_MODEL})")
    return llm


# ============================================================
# Document Preparation and Atlas Upload
# ============================================================

def prepare_documents() -> List[Document]:
    """Convert knowledge base data to LangChain Document format."""
    docs = [
        Document(
            page_content=item["text"],
            metadata={
                "title": item["title"],
                "category": item["category"],
                "source": item["source"],
            },
        )
        for item in MONGODB_KNOWLEDGE_BASE
    ]
    print(f"✅ {len(docs)} documents prepared")
    return docs


def load_documents_to_atlas(
    collection: pymongo.collection.Collection,
    embeddings: VoyageAIEmbeddings,
    docs: List[Document],
    force_reload: bool = False,
) -> None:
    """
    Upload documents with embeddings to MongoDB Atlas.
    Embeddings are stored in MongoDB for FAISS index rebuilding on next run.
    """
    existing_count = collection.count_documents({})

    if not force_reload and existing_count > 0:
        print(f"✅ Using existing data ({existing_count} documents)")
        return

    print(f"📥 Uploading {len(docs)} documents... (generating embeddings, may take a moment)")
    collection.drop()

    MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=embeddings,
        collection=collection,
        index_name=SEARCH_INDEX_NAME,  # used only as a label here
    )
    print(f"✅ Upload complete ({collection.count_documents({})} documents)")


# ============================================================
# FAISS Local Vector Store
# ============================================================

class FAISSVectorStore:
    """
    Local in-memory vector store using FAISS.

    Why FAISS instead of MongoDB $vectorSearch?
    - $vectorSearch requires Atlas M10+ cluster
    - FAISS runs locally, works with Free Tier (M0) and any cluster
    - Embeddings are still stored in MongoDB for persistence

    Similarity: cosine (via L2-normalized inner product)
    """

    def __init__(self, dims: int = EMBEDDING_DIMS):
        # IndexFlatIP: exact inner product search
        # After L2 normalization, inner product == cosine similarity
        self.index = faiss.IndexFlatIP(dims)
        self._doc_ids: List[str] = []
        self._docs: List[Dict] = []

    def add_documents(
        self,
        embeddings: List[List[float]],
        doc_ids: List[str],
        docs: List[Dict],
    ) -> None:
        """Add document embeddings to the FAISS index."""
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)   # normalize → inner product = cosine similarity
        self.index.add(vectors)
        self._doc_ids.extend(doc_ids)
        self._docs.extend(docs)

    def search(self, query_embedding: List[float], k: int = 10) -> List[Dict]:
        """Return the k most similar documents."""
        if self.index.ntotal == 0:
            return []
        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:   # FAISS pads with -1 when results < k
                continue
            doc = dict(self._docs[idx])
            doc["vector_score"] = float(score)
            results.append(doc)
        return results

    @property
    def total(self) -> int:
        return self.index.ntotal


def build_faiss_index(
    collection: pymongo.collection.Collection,
) -> FAISSVectorStore:
    """
    Build a FAISS index by reading embeddings stored in MongoDB.

    Embeddings were saved by langchain-mongodb when documents were uploaded.
    No additional API calls needed — we reuse stored vectors.
    """
    print("  Building FAISS index from stored embeddings...")

    docs = list(collection.find(
        {},
        {"_id": 1, TEXT_FIELD: 1, "title": 1, "category": 1, "source": 1, EMBEDDING_FIELD: 1},
    ))

    if not docs:
        raise ValueError("No documents in collection. Upload documents first.")

    # Extract embeddings (remove from doc dict to avoid passing large vectors around)
    embeddings = [doc.pop(EMBEDDING_FIELD) for doc in docs]
    doc_ids    = [str(doc["_id"]) for doc in docs]

    faiss_store = FAISSVectorStore(dims=EMBEDDING_DIMS)
    faiss_store.add_documents(embeddings, doc_ids, docs)

    print(f"  ✅ FAISS index built ({faiss_store.total} documents)")
    return faiss_store


# ============================================================
# Atlas Search Index (Full-Text, works on M0)
# ============================================================

def create_search_index(
    collection: pymongo.collection.Collection,
    force_recreate: bool = False,
) -> bool:
    """
    Create Atlas Search index for full-text search.

    Note: This is the $search (keyword) index, NOT $vectorSearch.
    Atlas Search works on MongoDB Atlas Free Tier (M0).
    Vector search is handled by local FAISS instead.
    """
    try:
        existing = {idx["name"] for idx in collection.list_search_indexes()}
    except Exception:
        existing = set()

    if SEARCH_INDEX_NAME in existing:
        if force_recreate:
            print(f"  Dropping existing index: {SEARCH_INDEX_NAME}")
            collection.drop_search_index(SEARCH_INDEX_NAME)
            _wait_for_index_drop(collection, SEARCH_INDEX_NAME)
        else:
            print(f"  ✅ Already exists: {SEARCH_INDEX_NAME}")
            return False

    search_index = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": False,
                "fields": {
                    TEXT_FIELD: {"type": "string"},
                    "title":    {"type": "string"},
                },
            }
        },
        name=SEARCH_INDEX_NAME,
        type="search",
    )

    print(f"  Requesting index creation: [{SEARCH_INDEX_NAME}]")
    collection.create_search_indexes([search_index])
    return True


def _wait_for_index_drop(
    collection: pymongo.collection.Collection,
    index_name: str,
    timeout: int = 120,
    poll_interval: int = 3,
) -> None:
    """Wait until the index is fully dropped (internal helper)."""
    start = time.time()
    while time.time() - start < timeout:
        names = {idx["name"] for idx in collection.list_search_indexes()}
        if index_name not in names:
            return
        time.sleep(poll_interval)


def wait_for_search_index_ready(
    collection: pymongo.collection.Collection,
    timeout: int = 300,
    poll_interval: int = 5,
) -> bool:
    """Poll until the Atlas Search index reaches READY status."""
    start = time.time()
    print(f"  Waiting for '{SEARCH_INDEX_NAME}' to become READY (max {timeout}s)...")

    while time.time() - start < timeout:
        for idx in collection.list_search_indexes():
            if idx["name"] == SEARCH_INDEX_NAME:
                status  = idx.get("status", "UNKNOWN")
                elapsed = int(time.time() - start)
                print(f"  [{elapsed:>3}s] {SEARCH_INDEX_NAME}: {status}", end="\r", flush=True)
                if status == "READY":
                    print(f"\n  ✅ Index READY! ({elapsed}s elapsed)")
                    return True
        time.sleep(poll_interval)

    print(f"\n  ⚠️  Timeout ({timeout}s)")
    return False


def setup_search_index(
    collection: pymongo.collection.Collection,
    force_recreate: bool = False,
    wait_timeout: int = 300,
) -> None:
    """Create Atlas Search index and wait for READY status."""
    print("\n[Atlas Search Index Setup]")
    created = create_search_index(collection, force_recreate=force_recreate)
    if created:
        wait_for_search_index_ready(collection, timeout=wait_timeout)
    else:
        print("  → Index already exists. Skipping.")


# ============================================================
# Hybrid Search: FAISS (Vector) + MongoDB $search (Full-Text)
# ============================================================

def vector_search(
    faiss_store: FAISSVectorStore,
    query_embedding: List[float],
    k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Semantic search using local FAISS index.

    Replaces MongoDB $vectorSearch — works with Atlas Free Tier (M0).
    Embeddings are loaded from MongoDB at startup into FAISS memory.
    """
    return faiss_store.search(query_embedding, k)


def text_search(
    collection: pymongo.collection.Collection,
    query: str,
    k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Full-text search using MongoDB Atlas $search.
    Works on Atlas Free Tier (M0).
    """
    pipeline = [
        {
            "$search": {
                "index": SEARCH_INDEX_NAME,
                "text": {
                    "query": query,
                    "path": [TEXT_FIELD, "title"],
                },
            }
        },
        {"$limit": k},
        {
            "$project": {
                "_id": 1,
                TEXT_FIELD: 1,
                "title": 1,
                "category": 1,
                "source": 1,
                "text_score": {"$meta": "searchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))


# ============================================================
# RRF (Reciprocal Rank Fusion)
# ============================================================

def reciprocal_rank_fusion(
    vector_results: List[Dict],
    text_results: List[Dict],
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Merge FAISS vector search and Atlas full-text search results using RRF.

    RRF Formula:
        RRF_score(d) = Σ  1 / (k + rank_i(d))

    Args:
        vector_results: FAISS search results
        text_results  : Atlas $search results
        rrf_k         : RRF constant k (default 60)

    Returns:
        Merged results sorted by RRF score (descending)
    """
    rrf_map: Dict[str, Dict] = {}

    for rank, doc in enumerate(vector_results, start=1):
        doc_id = str(doc.get("_id", id(doc)))
        if doc_id not in rrf_map:
            rrf_map[doc_id] = {
                "doc": doc, "rrf_score": 0.0,
                "vector_rank": None, "text_rank": None,
                "vector_score": None, "text_score": None,
            }
        rrf_map[doc_id]["rrf_score"]   += 1.0 / (rrf_k + rank)
        rrf_map[doc_id]["vector_rank"]  = rank
        rrf_map[doc_id]["vector_score"] = doc.get("vector_score")

    for rank, doc in enumerate(text_results, start=1):
        doc_id = str(doc.get("_id", id(doc)))
        if doc_id not in rrf_map:
            rrf_map[doc_id] = {
                "doc": doc, "rrf_score": 0.0,
                "vector_rank": None, "text_rank": None,
                "vector_score": None, "text_score": None,
            }
        rrf_map[doc_id]["rrf_score"]  += 1.0 / (rrf_k + rank)
        rrf_map[doc_id]["text_rank"]   = rank
        rrf_map[doc_id]["text_score"]  = doc.get("text_score")

    return sorted(rrf_map.values(), key=lambda x: x["rrf_score"], reverse=True)


def print_rrf_results(rrf_results: List[Dict], top_k: int = 5) -> None:
    """Print the RRF results table."""
    SEP  = "=" * 75
    DASH = "-" * 75
    print("\n" + SEP)
    print("  RRF (Reciprocal Rank Fusion) Search Results")
    print(SEP)
    COL1, COL2, COL3, COL4, COL5 = "Rank", "Document Title", "FAISS", "Atlas$search", "RRF Score"
    print(f"  {COL1:<4} {COL2:<32} {COL3:<6} {COL4:<13} {COL5:<12} Category")
    print(DASH)

    for i, result in enumerate(rrf_results[:top_k], start=1):
        doc      = result["doc"]
        title    = (doc.get("title")    or "Untitled")[:30]
        category = (doc.get("category") or "-")
        v_rank   = str(result["vector_rank"]) if result["vector_rank"] else "-"
        t_rank   = str(result["text_rank"])   if result["text_rank"]   else "-"
        print(f"  {i:<4} {title:<32} {v_rank:<6} {t_rank:<13} {result['rrf_score']:.6f}   {category}")

    print(SEP)
    print("\n  [Score Details]")
    for i, result in enumerate(rrf_results[:top_k], start=1):
        doc    = result["doc"]
        title  = (doc.get("title") or "Untitled")[:25]
        v_str  = f"{result['vector_score']:.4f}" if result["vector_score"] is not None else "N/A"
        t_str  = f"{result['text_score']:.4f}"   if result["text_score"]   is not None else "N/A"
        print(f"  {i}. {title}: FAISS={v_str}, Atlas$search={t_str}, RRF={result['rrf_score']:.6f}")
    print()


def hybrid_search(
    faiss_store: FAISSVectorStore,
    collection: pymongo.collection.Collection,
    query: str,
    embeddings: VoyageAIEmbeddings,
    k: int = 10,
    rrf_k: int = 60,
    verbose: bool = True,
) -> List[Dict]:
    """
    Hybrid search: FAISS vector search + MongoDB Atlas full-text search + RRF.

    Args:
        faiss_store: Local FAISS index (semantic search)
        collection : MongoDB collection (full-text search via $search)
        query      : Search query string
        embeddings : Voyage AI embedding model
        k          : Number of results per search
        rrf_k      : RRF constant k
        verbose    : Whether to print detailed results
    """
    # 1. Generate query embedding
    query_embedding = embeddings.embed_query(query)

    # 2. FAISS vector search (semantic)
    v_results = vector_search(faiss_store, query_embedding, k=k)

    # 3. MongoDB Atlas full-text search (keyword)
    t_results = text_search(collection, query, k=k)

    if verbose:
        print(f"  → FAISS vector: {len(v_results)} results  |  Atlas $search: {len(t_results)} results")

    # 4. RRF fusion
    rrf_results = reciprocal_rank_fusion(v_results, t_results, rrf_k=rrf_k)

    if verbose:
        print_rrf_results(rrf_results, top_k=5)

    return rrf_results


# ============================================================
# RAG Chain
# ============================================================

def format_context(rrf_results: List[Dict], top_k: int = 3) -> str:
    """Build RAG context from the top RRF results."""
    parts = []
    for i, result in enumerate(rrf_results[:top_k], start=1):
        doc     = result["doc"]
        title   = doc.get("title", "")
        content = doc.get(TEXT_FIELD, "")
        parts.append(
            f"--- Document {i}: {title} (RRF Score: {result['rrf_score']:.4f}) ---\n{content}"
        )
    return "\n\n".join(parts)


def build_rag_chain(llm: ChatOpenAI):
    """Build a RAG chain using LangChain LCEL."""
    prompt = ChatPromptTemplate.from_template(
        """You are a MongoDB expert.
Using the provided context, give accurate and practical answers to MongoDB troubleshooting questions.

[Reference Documents]
{context}

[Question]
{question}

[Answer Guidelines]
1. Prioritize information from the reference documents.
2. Include specific solutions and commands.
3. Explain step by step: immediate action → root cause resolution → prevention.

[Answer]
"""
    )
    return prompt | llm | StrOutputParser()


def ask_mongodb_question(
    question: str,
    faiss_store: FAISSVectorStore,
    collection: pymongo.collection.Collection,
    embeddings: VoyageAIEmbeddings,
    llm: ChatOpenAI,
    context_top_k: int = 3,
    search_k: int = 10,
    rrf_k: int = 60,
    verbose: bool = True,
) -> str:
    """Answer a MongoDB troubleshooting question using RAG + Hybrid Search."""
    SEP = "=" * 75
    print(f"\n{SEP}")
    print(f"  Question: {question}")
    print(SEP)

    print("\n[Searching...]")
    rrf_results = hybrid_search(
        faiss_store=faiss_store,
        collection=collection,
        query=question,
        embeddings=embeddings,
        k=search_k,
        rrf_k=rrf_k,
        verbose=verbose,
    )

    context = format_context(rrf_results, top_k=context_top_k)

    print("[Generating answer...]")
    answer = build_rag_chain(llm).invoke({"context": context, "question": question})

    DASH = "-" * 75
    print(f"\n{DASH}")
    print("  [Answer]")
    print(DASH)
    print(answer)
    print(f"{SEP}\n")
    return answer


# ============================================================
# Main
# ============================================================

def main():
    """Main execution function."""

    # ── 1. Initialize components ────────────────────────────────
    print("\n" + "="*55)
    print(" [1/5] Initializing components")
    print("="*55)
    client     = get_mongodb_client()
    embeddings = get_embeddings()
    llm        = get_llm()
    collection = client[DB_NAME][COLLECTION_NAME]

    # ── 2. Upload documents to MongoDB ──────────────────────────
    print("\n" + "="*55)
    print(" [2/5] Loading knowledge base documents")
    print("="*55)
    docs = prepare_documents()
    load_documents_to_atlas(collection, embeddings, docs, force_reload=False)

    # ── 3. Build FAISS index from stored embeddings ─────────────
    print("\n" + "="*55)
    print(" [3/5] Building local FAISS vector index")
    print("="*55)
    faiss_store = build_faiss_index(collection)

    # ── 4. Setup Atlas Search index (full-text) ──────────────────
    print("\n" + "="*55)
    print(" [4/5] Setting up Atlas Search index (full-text)")
    print("="*55)
    setup_search_index(collection, force_recreate=False, wait_timeout=300)

    # ── 5. Run sample questions ─────────────────────────────────
    print("\n" + "="*55)
    print(" [5/5] Running sample questions")
    print("="*55)
    sample_questions = [
        "How do I resolve MongoDB connection pool exhaustion?",
        "What are the causes and solutions for replication lag?",
        "How do I optimize slow queries in MongoDB?",
    ]
    for question in sample_questions:
        ask_mongodb_question(
            question=question,
            faiss_store=faiss_store,
            collection=collection,
            embeddings=embeddings,
            llm=llm,
            context_top_k=3,
            verbose=True,
        )
        time.sleep(1)

    # ── Interactive mode ─────────────────────────────────────────
    print("\n" + "="*55)
    print(" Interactive Q&A Mode (type 'q' to quit)")
    print("="*55)
    while True:
        try:
            question = input("\nQuestion > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if question.lower() in {"q", "quit", "exit"}:
            print("Exiting.")
            break
        if question:
            ask_mongodb_question(
                question=question,
                faiss_store=faiss_store,
                collection=collection,
                embeddings=embeddings,
                llm=llm,
                context_top_k=3,
                verbose=True,
            )

    client.close()


if __name__ == "__main__":
    main()
