"""
MongoDB Troubleshooting RAG (Retrieval-Augmented Generation) System
===================================================================
Tech Stack:
  - LangChain    : RAG pipeline construction
  - Voyage AI    : voyage-4 text embeddings (1024 dimensions)
  - MongoDB Atlas: Vector store + hybrid search
  - OpenAI       : GPT-4o-mini answer generation
  - Hybrid Search: Vector search (Semantic) + Full-text search (Keyword)
  - RRF          : Reciprocal Rank Fusion result merging

Run:
  python mongodb_rag.py
"""

import os
import time
from typing import List, Dict, Any, Optional

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
VECTOR_INDEX_NAME = "vector_index"   # Atlas Vector Search index name
SEARCH_INDEX_NAME = "search_index"   # Atlas Full-Text Search index name
TEXT_FIELD        = "text"
EMBEDDING_FIELD   = "embedding"

# Model settings
EMBEDDING_MODEL = "voyage-4"         # voyage-4 (1024 dimensions)
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
            "- Set Oplog size to cover at least 24-72 hours of operations\n"
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
            "- Monitor cache hit ratio continuously (target: 95%+)\n"
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
            "1. Wait for automatic election after network recovery (typically 10-30 seconds)\n\n"
            "2. Adjust member priority to guide Primary election\n"
            "   cfg = rs.conf()\n"
            "   cfg.members[0].priority = 2\n"
            "   rs.reconfig(cfg)\n\n"
            "Prevention:\n"
            "- Always maintain an odd number of voting members (3, 5, 7)\n"
            "- Deploy members across different Availability Zones"
        ),
    },
    {
        "title": "Atlas Vector Search Index Error",
        "category": "search",
        "source": "MongoDB Atlas Search Guide",
        "text": (
            "MongoDB Atlas Vector Search Index Error Troubleshooting Guide\n\n"
            "Symptoms:\n"
            "- '$vectorSearch is not allowed' error\n"
            "- Empty array returned from search\n"
            "- Index status shows FAILED or BUILDING\n\n"
            "Causes:\n"
            "- Atlas Search index not created\n"
            "- Index name mismatch\n"
            "- numDimensions does not match embedding model dimensions\n\n"
            "Solutions:\n"
            "1. Verify Vector Search index settings\n"
            "   numDimensions: 1024  (for voyage-4 model)\n\n"
            "2. Verify index name matches in code\n\n"
            "3. Wait for index rebuild (large collections may take several minutes)\n\n"
            "Prevention:\n"
            "- Manage index names as code constants"
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
) -> MongoDBAtlasVectorSearch:
    """Upload documents to MongoDB Atlas and return the vector store."""
    existing_count = collection.count_documents({})

    if not force_reload and existing_count > 0:
        print(f"✅ Using existing data ({existing_count} documents)")
    else:
        print(f"📥 Uploading {len(docs)} documents to MongoDB Atlas... (may take a moment)")
        collection.drop()
        MongoDBAtlasVectorSearch.from_documents(
            documents=docs,
            embedding=embeddings,
            collection=collection,
            index_name=VECTOR_INDEX_NAME,
        )
        print(f"✅ Upload complete ({collection.count_documents({})} documents)")

    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=VECTOR_INDEX_NAME,
        text_key=TEXT_FIELD,
        embedding_key=EMBEDDING_FIELD,
    )


# ============================================================
# Atlas Index Auto-Creation (pymongo 4.7+ SearchIndexModel)
# ============================================================

def create_atlas_indexes(
    collection: pymongo.collection.Collection,
    force_recreate: bool = False,
) -> bool:
    """
    Automatically create Vector Search and Atlas Search indexes from code.
    Uses pymongo 4.7+ SearchIndexModel / create_search_indexes() API.

    Args:
        collection    : MongoDB collection
        force_recreate: If True, drops and recreates existing indexes

    Returns:
        True if new indexes were created
    """
    try:
        existing = {idx["name"] for idx in collection.list_search_indexes()}
    except Exception:
        existing = set()

    to_create: List[SearchIndexModel] = []

    # ── [1] Vector Search Index ────────────────────────────────
    if VECTOR_INDEX_NAME in existing:
        if force_recreate:
            print(f"  Dropping existing index: {VECTOR_INDEX_NAME}")
            collection.drop_search_index(VECTOR_INDEX_NAME)
            _wait_for_index_drop(collection, VECTOR_INDEX_NAME)
        else:
            print(f"  ✅ Already exists: {VECTOR_INDEX_NAME}")

    if VECTOR_INDEX_NAME not in existing or force_recreate:
        to_create.append(
            SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": EMBEDDING_FIELD,
                            "numDimensions": EMBEDDING_DIMS,  # voyage-4: 1024 dims
                            "similarity": "cosine",
                        }
                    ]
                },
                name=VECTOR_INDEX_NAME,
                type="vectorSearch",
            )
        )

    # ── [2] Atlas Search (Full-Text) Index ────────────────────
    if SEARCH_INDEX_NAME in existing:
        if force_recreate:
            print(f"  Dropping existing index: {SEARCH_INDEX_NAME}")
            collection.drop_search_index(SEARCH_INDEX_NAME)
            _wait_for_index_drop(collection, SEARCH_INDEX_NAME)
        else:
            print(f"  ✅ Already exists: {SEARCH_INDEX_NAME}")

    if SEARCH_INDEX_NAME not in existing or force_recreate:
        to_create.append(
            SearchIndexModel(
                definition={
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            TEXT_FIELD: {"type": "string"},
                            "title":    {"type": "string"},  # top-level title field
                        },
                    }
                },
                name=SEARCH_INDEX_NAME,
                type="search",
            )
        )

    if to_create:
        names = [m.document["name"] for m in to_create]
        print(f"  Requesting index creation: {names}")
        collection.create_search_indexes(to_create)
        return True

    return False


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


def wait_for_indexes_ready(
    collection: pymongo.collection.Collection,
    index_names: List[str],
    timeout: int = 300,
    poll_interval: int = 5,
) -> bool:
    """
    Poll until all specified indexes reach 'READY' status.

    Args:
        collection   : MongoDB collection
        index_names  : List of index names to wait for
        timeout      : Maximum wait time in seconds
        poll_interval: Polling interval in seconds

    Returns:
        True if all indexes are READY, False if timed out
    """
    target = set(index_names)
    start  = time.time()
    print(f"  Waiting for indexes to become READY (max {timeout}s)...")

    while time.time() - start < timeout:
        ready, parts = set(), []
        for idx in collection.list_search_indexes():
            if idx["name"] in target:
                status = idx.get("status", "UNKNOWN")
                parts.append(f"{idx['name']}: {status}")
                if status == "READY":
                    ready.add(idx["name"])

        elapsed = int(time.time() - start)
        print(f"  [{elapsed:>3}s] {' | '.join(parts)}", end="\r", flush=True)

        if ready == target:
            print(f"\n  ✅ All indexes READY! ({elapsed}s elapsed)")
            return True

        time.sleep(poll_interval)

    print(f"\n  ⚠️  Timeout ({timeout}s): some indexes are not ready yet.")
    return False


def setup_indexes(
    collection: pymongo.collection.Collection,
    force_recreate: bool = False,
    wait_timeout: int = 300,
) -> None:
    """Convenience function: create indexes and wait for READY status."""
    print("\n[Index Setup]")
    created = create_atlas_indexes(collection, force_recreate=force_recreate)
    if created:
        wait_for_indexes_ready(
            collection,
            index_names=[VECTOR_INDEX_NAME, SEARCH_INDEX_NAME],
            timeout=wait_timeout,
        )
    else:
        print("  → All indexes already exist. Skipping.")


# ============================================================
# Hybrid Search: Vector Search + Full-Text Search
# ============================================================

def vector_search(
    collection: pymongo.collection.Collection,
    query_embedding: List[float],
    k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Semantic search using $vectorSearch.

    Note: langchain-mongodb stores metadata fields at the top level of the document.
    e.g., {"text": ..., "embedding": [...], "title": ..., "category": ...}
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": EMBEDDING_FIELD,
                "queryVector": query_embedding,
                "numCandidates": k * 10,  # candidate pool = k * 10 recommended
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 1,
                TEXT_FIELD: 1,
                "title": 1,       # top-level field
                "category": 1,
                "source": 1,
                "vector_score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))


def text_search(
    collection: pymongo.collection.Collection,
    query: str,
    k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Full-text search using $search (Atlas Search).
    Requires the Atlas Search index (search_index) to be created.
    """
    pipeline = [
        {
            "$search": {
                "index": SEARCH_INDEX_NAME,
                "text": {
                    "query": query,
                    "path": [TEXT_FIELD, "title"],  # top-level fields
                },
            }
        },
        {"$limit": k},
        {
            "$project": {
                "_id": 1,
                TEXT_FIELD: 1,
                "title": 1,       # top-level field
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
    Merge vector search and full-text search results using RRF.

    RRF Formula:
        RRF_score(d) = Σ  1 / (k + rank_i(d))

    Args:
        vector_results: Vector search results (with ranks)
        text_results  : Full-text search results (with ranks)
        rrf_k         : RRF constant k (default 60)

    Returns:
        Merged results sorted by RRF score (descending)
    """
    rrf_map: Dict[str, Dict] = {}

    for rank, doc in enumerate(vector_results, start=1):
        doc_id = str(doc["_id"])
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
        doc_id = str(doc["_id"])
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
    print("\n" + "=" * 75)
    print("  RRF (Reciprocal Rank Fusion) Search Results")
    print("=" * 75)
    print(f"  {'Rank':<4} {'Document Title':<32} {'VecRank':<8} {'TxtRank':<10} {'RRF Score':<12} Category")
    print("-" * 75)

    for i, result in enumerate(rrf_results[:top_k], start=1):
        doc      = result["doc"]
        title    = (doc.get("title")    or "Untitled")[:30]
        category = (doc.get("category") or "-")
        v_rank   = str(result["vector_rank"]) if result["vector_rank"] else "-"
        t_rank   = str(result["text_rank"])   if result["text_rank"]   else "-"
        rrf_score = result["rrf_score"]
        print(f"  {i:<4} {title:<32} {v_rank:<8} {t_rank:<10} {rrf_score:.6f}   {category}")

    print("=" * 75)
    print("\n  [Score Details]")
    for i, result in enumerate(rrf_results[:top_k], start=1):
        doc     = result["doc"]
        title   = (doc.get("title") or "Untitled")[:25]
        v_str   = f"{result['vector_score']:.4f}" if result["vector_score"] is not None else "N/A"
        t_str   = f"{result['text_score']:.4f}"   if result["text_score"]   is not None else "N/A"
        print(f"  {i}. {title}: vector={v_str}, text={t_str}, RRF={result['rrf_score']:.6f}")
    print()


def hybrid_search(
    collection: pymongo.collection.Collection,
    query: str,
    embeddings: VoyageAIEmbeddings,
    k: int = 10,
    rrf_k: int = 60,
    verbose: bool = True,
) -> List[Dict]:
    """Hybrid search: Vector Search + Full-Text Search + RRF fusion."""
    query_embedding = embeddings.embed_query(query)
    v_results = vector_search(collection, query_embedding, k=k)
    t_results = text_search(collection, query, k=k)

    if verbose:
        print(f"  → Vector search: {len(v_results)} results  |  Full-text search: {len(t_results)} results")

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
    collection: pymongo.collection.Collection,
    embeddings: VoyageAIEmbeddings,
    llm: ChatOpenAI,
    context_top_k: int = 3,
    search_k: int = 10,
    rrf_k: int = 60,
    verbose: bool = True,
) -> str:
    """Answer a MongoDB troubleshooting question using RAG."""
    print(f"\n{'='*75}")
    print(f"  Question: {question}")
    print(f"{'='*75}")

    print("\n[Searching...]")
    rrf_results = hybrid_search(
        collection=collection, query=question, embeddings=embeddings,
        k=search_k, rrf_k=rrf_k, verbose=verbose,
    )

    context = format_context(rrf_results, top_k=context_top_k)

    print("[Generating answer...]")
    answer = build_rag_chain(llm).invoke({"context": context, "question": question})

    print(f"\n{'─'*75}")
    print("  [Answer]")
    print(f"{'─'*75}")
    print(answer)
    print(f"{'='*75}\n")
    return answer


# ============================================================
# Main
# ============================================================

def main():
    """Main execution function."""

    # ── 1. Initialize components ────────────────────────────────
    print("\n" + "="*50)
    print(" [1/5] Initializing components")
    print("="*50)
    client     = get_mongodb_client()
    embeddings = get_embeddings()
    llm        = get_llm()
    collection = client[DB_NAME][COLLECTION_NAME]

    # ── 2. Load documents ───────────────────────────────────────
    print("\n" + "="*50)
    print(" [2/5] Loading knowledge base documents")
    print("="*50)
    docs = prepare_documents()
    load_documents_to_atlas(collection, embeddings, docs, force_reload=False)

    # ── 3. Auto-create indexes ──────────────────────────────────
    print("\n" + "="*50)
    print(" [3/5] Setting up Atlas indexes")
    print("="*50)
    setup_indexes(collection, force_recreate=False, wait_timeout=300)

    # ── 4. Run sample questions ─────────────────────────────────
    print("\n" + "="*50)
    print(" [4/5] Running sample questions")
    print("="*50)
    sample_questions = [
        "How do I resolve MongoDB connection pool exhaustion?",
        "What are the causes and solutions for replication lag?",
        "How do I optimize slow queries in MongoDB?",
    ]
    for question in sample_questions:
        ask_mongodb_question(
            question=question, collection=collection,
            embeddings=embeddings, llm=llm, context_top_k=3, verbose=True,
        )
        time.sleep(1)

    # ── 5. Interactive mode ─────────────────────────────────────
    print("\n" + "="*50)
    print(" [5/5] Interactive Q&A Mode")
    print("="*50)
    print("Ask any MongoDB troubleshooting question. (type 'q' to quit)\n")
    while True:
        try:
            question = input("Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if question.lower() in {"q", "quit", "exit"}:
            print("Exiting.")
            break
        if question:
            ask_mongodb_question(
                question=question, collection=collection,
                embeddings=embeddings, llm=llm, context_top_k=3, verbose=True,
            )

    client.close()


if __name__ == "__main__":
    main()
