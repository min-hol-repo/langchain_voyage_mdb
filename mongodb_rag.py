"""
MongoDB 장애 대응 RAG (Retrieval-Augmented Generation) 시스템
=============================================================
사용 기술:
  - LangChain    : RAG 파이프라인 구성
  - Voyage AI    : voyage-4 텍스트 임베딩 (1024차원)
  - MongoDB Atlas: 벡터 저장소 + 하이브리드 검색
  - OpenAI       : GPT-4o-mini 답변 생성
  - 하이브리드 검색: 벡터 검색(Semantic) + 전문 검색(Full-Text)
  - RRF          : Reciprocal Rank Fusion 결과 융합

실행:
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
# 환경 변수 로드
# ============================================================
load_dotenv()

MONGODB_URI    = os.getenv("MONGODB_URI", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")

# MongoDB 설정
DB_NAME           = "mongodb_troubleshooting"
COLLECTION_NAME   = "knowledge_base"
VECTOR_INDEX_NAME = "vector_index"   # Atlas 벡터 검색 인덱스 이름
SEARCH_INDEX_NAME = "search_index"   # Atlas 전문 검색 인덱스 이름
TEXT_FIELD        = "text"
EMBEDDING_FIELD   = "embedding"

# 모델 설정
EMBEDDING_MODEL = "voyage-4"         # voyage-4 (1024차원)
EMBEDDING_DIMS  = 1024
LLM_MODEL       = "gpt-4o-mini"


# ============================================================
# MongoDB 장애 대응 지식 베이스 (샘플 데이터)
# ============================================================
MONGODB_KNOWLEDGE_BASE = [
    {
        "title": "Connection Pool 고갈 문제",
        "category": "connection",
        "source": "MongoDB 연결 관리 가이드",
        "text": (
            "MongoDB Connection Pool 고갈 문제 해결 가이드\n\n"
            "증상:\n"
            "- 'connection pool exhausted' 또는 'too many connections' 오류 발생\n"
            "- 애플리케이션 응답 지연 및 타임아웃\n"
            "- mongostat에서 connections 수가 maxIncomingConnections에 근접\n\n"
            "원인:\n"
            "- 커넥션을 반납하지 않는 코드 (커넥션 누수)\n"
            "- 갑작스러운 트래픽 증가\n"
            "- maxPoolSize 설정이 너무 낮음\n"
            "- 슬로우 쿼리로 인한 커넥션 점유 시간 증가\n\n"
            "해결 방법:\n"
            "1. 현재 커넥션 상태 확인\n"
            "   db.serverStatus().connections\n\n"
            "2. maxPoolSize 조정 (드라이버 설정)\n"
            "   MongoClient(uri, maxPoolSize=100, waitQueueTimeoutMS=5000)\n\n"
            "3. 커넥션 누수 확인\n"
            "   db.currentOp()로 장시간 열려있는 세션 확인\n\n"
            "예방:\n"
            "- 커넥션 풀 모니터링 (Atlas Metrics > Connections)\n"
            "- 알림 설정 (connections > 임계값)\n"
            "- 슬로우 쿼리 최적화로 커넥션 점유 시간 단축"
        ),
    },
    {
        "title": "인덱스 누락으로 인한 슬로우 쿼리",
        "category": "performance",
        "source": "MongoDB 성능 최적화 가이드",
        "text": (
            "인덱스 누락으로 인한 슬로우 쿼리 해결 가이드\n\n"
            "증상:\n"
            "- 쿼리 응답 시간이 수 초 이상\n"
            "- Atlas Profiler에 COLLSCAN(전체 스캔) 표시\n"
            "- CPU 사용률 급상승\n\n"
            "진단 방법:\n"
            "1. 슬로우 쿼리 로그 확인\n"
            "   db.system.profile.find({millis: {$gt: 100}}).sort({ts: -1}).limit(10)\n\n"
            "2. explain()으로 실행 계획 분석\n"
            "   db.collection.find(query).explain('executionStats')\n\n"
            "해결 방법:\n"
            "1. 필요한 인덱스 생성\n"
            "   db.orders.createIndex({user_id: 1, created_at: -1})\n\n"
            "2. 복합 인덱스 설계 원칙 (ESR 규칙)\n"
            "   - Equality(동등 조건) 필드 먼저\n"
            "   - Sort(정렬) 필드 다음\n"
            "   - Range(범위 조건) 필드 마지막\n\n"
            "예방:\n"
            "- 새 쿼리 패턴 도입 시 인덱스 계획 수립\n"
            "- Atlas Performance Advisor 권장 인덱스 모니터링"
        ),
    },
    {
        "title": "Replication Lag 복제 지연 문제",
        "category": "replication",
        "source": "MongoDB 복제 운영 가이드",
        "text": (
            "MongoDB Replication Lag(복제 지연) 해결 가이드\n\n"
            "증상:\n"
            "- rs.printSlaveReplicationInfo()에서 지연 시간이 큰 경우\n"
            "- Secondary에서 읽기 시 오래된 데이터 반환\n\n"
            "원인:\n"
            "- Primary에 과도한 쓰기 부하\n"
            "- Secondary 서버의 리소스 부족 (CPU, I/O)\n"
            "- 네트워크 대역폭 부족\n"
            "- Oplog 크기 부족으로 인한 롤백\n\n"
            "진단 방법:\n"
            "1. 복제 상태 확인\n"
            "   rs.status()\n"
            "   rs.printSlaveReplicationInfo()\n\n"
            "해결 방법:\n"
            "1. 즉각적 조치 - 부하 분산\n"
            "   - 읽기 요청을 Primary로 임시 전환\n"
            "   - 배치 작업 일시 중지\n\n"
            "2. Oplog 크기 증설\n"
            "   mongod.conf: replication.oplogSizeMB: 51200\n\n"
            "예방:\n"
            "- Oplog 크기를 최소 24-72시간 분량으로 설정\n"
            "- Secondary 서버 사양을 Primary와 동일하게 유지"
        ),
    },
    {
        "title": "WiredTiger 캐시 메모리 부족",
        "category": "memory",
        "source": "MongoDB 메모리 관리 가이드",
        "text": (
            "WiredTiger 캐시 메모리 부족 문제 해결 가이드\n\n"
            "증상:\n"
            "- 쿼리 성능 급격히 저하\n"
            "- 디스크 I/O 급증\n"
            "- 'WiredTiger eviction' 경고 메시지\n"
            "- 서버 스왑(SWAP) 사용 증가\n\n"
            "원인:\n"
            "- 데이터셋이 WiredTiger 캐시 크기 초과\n"
            "- 메모리 설정이 서버 사양에 비해 낮게 설정됨\n\n"
            "해결 방법:\n"
            "1. WiredTiger 캐시 크기 증설\n"
            "   mongod.conf:\n"
            "     storage.wiredTiger.engineConfig.cacheSizeGB: 4  # RAM의 50% 권장\n\n"
            "2. 불필요한 인덱스 제거로 메모리 절약\n\n"
            "예방:\n"
            "- 캐시 히트율 지속 모니터링 (목표: 95% 이상)\n"
            "- Atlas 사용 시 적절한 인스턴스 티어 선택"
        ),
    },
    {
        "title": "Disk Space 디스크 공간 부족 장애",
        "category": "storage",
        "source": "MongoDB 스토리지 관리 가이드",
        "text": (
            "MongoDB Disk Space 부족 장애 해결 가이드\n\n"
            "증상:\n"
            "- 'no space left on device' 오류\n"
            "- 쓰기 작업 실패\n"
            "- MongoDB 프로세스 비정상 종료\n\n"
            "즉각적인 조치:\n"
            "1. 현재 디스크 사용량 확인\n"
            "   df -h  (OS 레벨)\n"
            "   db.stats()  (MongoDB 레벨)\n\n"
            "해결 방법:\n"
            "1. 오래된 데이터 삭제\n"
            "   db.logs.deleteMany({createdAt: {$lt: new Date('2024-01-01')}})\n\n"
            "2. compact 명령으로 공간 회수\n"
            "   db.runCommand({compact: 'collection_name'})\n\n"
            "3. TTL 인덱스로 자동 데이터 만료\n"
            "   db.logs.createIndex({createdAt: 1}, {expireAfterSeconds: 2592000})\n\n"
            "예방:\n"
            "- 디스크 사용량 80% 도달 시 알림 설정\n"
            "- Atlas Online Archive 활용"
        ),
    },
    {
        "title": "Lock 경합 (Lock Contention) 문제",
        "category": "locking",
        "source": "MongoDB 락 관리 가이드",
        "text": (
            "MongoDB Lock 경합(Lock Contention) 해결 가이드\n\n"
            "증상:\n"
            "- 전반적인 성능 저하\n"
            "- globalLock.ratio 값이 0.5 이상\n"
            "- 쿼리 대기 시간 증가\n\n"
            "진단 방법:\n"
            "1. 현재 실행 중인 작업 확인\n"
            "   db.currentOp({active: true, waitingForLock: true})\n\n"
            "해결 방법:\n"
            "1. 장시간 실행 중인 작업 중단\n"
            "   db.killOp(opid)\n\n"
            "2. 대용량 작업 배치 처리로 분할\n"
            "   - 한 번에 1000건씩 처리하고 sleep 추가\n\n"
            "예방:\n"
            "- 대용량 작업 시 배치 처리 패턴 적용\n"
            "- 모든 쿼리에 적절한 인덱스 확보"
        ),
    },
    {
        "title": "Primary 선출 실패 (Election Failure)",
        "category": "replication",
        "source": "MongoDB 복제셋 운영 가이드",
        "text": (
            "MongoDB Replica Set Primary 선출 실패 해결 가이드\n\n"
            "증상:\n"
            "- 모든 멤버가 SECONDARY 또는 UNKNOWN 상태\n"
            "- 쓰기 작업 불가 ('not master' 오류)\n"
            "- rs.status()에서 no primary 상태\n\n"
            "원인:\n"
            "- 네트워크 파티션 (스플릿 브레인)\n"
            "- 과반수(Majority) 멤버에 연결 불가\n\n"
            "해결 방법:\n"
            "1. 네트워크 복구 후 자동 선출 대기 (통상 10-30초)\n\n"
            "2. 특정 멤버 우선순위 조정\n"
            "   cfg = rs.conf()\n"
            "   cfg.members[0].priority = 2\n"
            "   rs.reconfig(cfg)\n\n"
            "예방:\n"
            "- 항상 홀수 개의 투표 멤버 유지 (3, 5, 7개)\n"
            "- 멤버를 서로 다른 AZ에 배포"
        ),
    },
    {
        "title": "Atlas Vector Search 인덱스 오류",
        "category": "search",
        "source": "MongoDB Atlas 검색 가이드",
        "text": (
            "MongoDB Atlas Vector Search 인덱스 오류 해결 가이드\n\n"
            "증상:\n"
            "- '$vectorSearch is not allowed' 오류\n"
            "- 검색 결과가 빈 배열 반환\n"
            "- 인덱스 상태가 FAILED 또는 BUILDING\n\n"
            "원인:\n"
            "- Atlas Search 인덱스가 생성되지 않음\n"
            "- 인덱스 이름 불일치\n"
            "- numDimensions가 임베딩 모델 차원수와 불일치\n\n"
            "해결 방법:\n"
            "1. Vector Search 인덱스 설정 확인\n"
            "   numDimensions: 1024  (voyage-4 모델)\n\n"
            "2. 인덱스 이름 코드와 일치 여부 확인\n\n"
            "3. 인덱스 재빌드 대기 (대용량 컬렉션은 수 분 소요)\n\n"
            "예방:\n"
            "- 인덱스 이름을 코드 상수로 관리"
        ),
    },
    {
        "title": "OOM (Out of Memory) 킬러 발생",
        "category": "memory",
        "source": "MongoDB 메모리 관리 가이드",
        "text": (
            "MongoDB OOM (Out of Memory) 킬러 발생 해결 가이드\n\n"
            "증상:\n"
            "- MongoDB 프로세스가 갑자기 종료됨\n"
            "- 'Killed process (mongod)' 로그\n"
            "- 시스템 메모리 사용량 100% 도달\n\n"
            "원인:\n"
            "- WiredTiger 캐시 + 인덱스가 가용 메모리 초과\n"
            "- 다른 프로세스와의 메모리 경쟁\n\n"
            "해결 방법:\n"
            "1. WiredTiger 캐시 크기 제한\n"
            "   cacheSizeGB: 2  (가용 메모리의 50% 이하)\n\n"
            "2. Swap 설정\n"
            "   fallocate -l 4G /swapfile && mkswap /swapfile && swapon /swapfile\n\n"
            "예방:\n"
            "- 메모리 사용률 85% 알림 설정\n"
            "- Atlas 사용 시 적절한 티어 선택"
        ),
    },
    {
        "title": "Mongodump / Mongorestore 백업 및 복구",
        "category": "backup",
        "source": "MongoDB 백업 복구 가이드",
        "text": (
            "MongoDB 백업 및 복구 완전 가이드\n\n"
            "Mongodump (백업):\n"
            "1. 전체 데이터베이스 백업\n"
            "   mongodump --uri='mongodb+srv://...' --out=/backup/$(date +%Y%m%d)\n\n"
            "2. 특정 컬렉션 백업\n"
            "   mongodump --uri='...' --db=mydb --collection=users --out=/backup/\n\n"
            "Mongorestore (복구):\n"
            "1. 전체 복구\n"
            "   mongorestore --uri='mongodb+srv://...' /backup/20240101/\n\n"
            "2. 병렬 복구로 속도 향상\n"
            "   mongorestore --uri='...' --numParallelCollections=4 /backup/\n\n"
            "Atlas 백업:\n"
            "- Atlas > Cluster > Backup > Take Snapshot\n"
            "- Point-in-Time Recovery (PITR) 활용\n"
            "- M10 이상에서 Continuous Cloud Backup 권장\n\n"
            "주의사항:\n"
            "- Replica Set에서 --oplog 옵션으로 일관성 확보\n"
            "- 대용량 백업 시 --gzip 옵션으로 압축"
        ),
    },
]


# ============================================================
# 컴포넌트 초기화
# ============================================================

def get_mongodb_client() -> MongoClient:
    """MongoDB Atlas에 연결합니다."""
    if not MONGODB_URI:
        raise ValueError("MONGODB_URI 환경 변수가 설정되지 않았습니다.")
    client = MongoClient(MONGODB_URI)
    client.admin.command("ping")
    print("✅ MongoDB Atlas 연결 성공")
    return client


def get_embeddings() -> VoyageAIEmbeddings:
    """Voyage AI 임베딩 모델을 초기화합니다."""
    if not VOYAGE_API_KEY:
        raise ValueError("VOYAGE_API_KEY 환경 변수가 설정되지 않았습니다.")
    emb = VoyageAIEmbeddings(voyage_api_key=VOYAGE_API_KEY, model=EMBEDDING_MODEL)
    print(f"✅ Voyage AI 임베딩 초기화 완료 (모델: {EMBEDDING_MODEL}, {EMBEDDING_DIMS}차원)")
    return emb


def get_llm() -> ChatOpenAI:
    """OpenAI LLM을 초기화합니다."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
    print(f"✅ OpenAI LLM 초기화 완료 (모델: {LLM_MODEL})")
    return llm


# ============================================================
# 문서 준비 및 Atlas 업로드
# ============================================================

def prepare_documents() -> List[Document]:
    """지식 베이스 데이터를 LangChain Document 형식으로 변환합니다."""
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
    print(f"✅ {len(docs)}개 문서 준비 완료")
    return docs


def load_documents_to_atlas(
    collection: pymongo.collection.Collection,
    embeddings: VoyageAIEmbeddings,
    docs: List[Document],
    force_reload: bool = False,
) -> MongoDBAtlasVectorSearch:
    """문서를 MongoDB Atlas에 업로드하고 벡터 스토어를 반환합니다."""
    existing_count = collection.count_documents({})

    if not force_reload and existing_count > 0:
        print(f"✅ 기존 데이터 사용 중 ({existing_count}개 문서)")
    else:
        print(f"📥 {len(docs)}개 문서 업로드 중... (임베딩 생성에 수십 초 소요)")
        collection.drop()
        MongoDBAtlasVectorSearch.from_documents(
            documents=docs,
            embedding=embeddings,
            collection=collection,
            index_name=VECTOR_INDEX_NAME,
        )
        print(f"✅ 업로드 완료 ({collection.count_documents({})}개)")

    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=VECTOR_INDEX_NAME,
        text_key=TEXT_FIELD,
        embedding_key=EMBEDDING_FIELD,
    )


# ============================================================
# Atlas 인덱스 자동 생성 (pymongo 4.7+ SearchIndexModel)
# ============================================================

def create_atlas_indexes(
    collection: pymongo.collection.Collection,
    force_recreate: bool = False,
) -> bool:
    """
    Vector Search 인덱스와 Atlas Search 인덱스를 코드에서 자동으로 생성합니다.

    Args:
        collection    : MongoDB 컬렉션
        force_recreate: True이면 기존 인덱스를 삭제하고 재생성

    Returns:
        새로 생성된 인덱스가 있으면 True
    """
    try:
        existing = {idx["name"] for idx in collection.list_search_indexes()}
    except Exception:
        existing = set()

    to_create: List[SearchIndexModel] = []

    # ── [1] Vector Search 인덱스 ──────────────────────────────
    if VECTOR_INDEX_NAME in existing:
        if force_recreate:
            print(f"  기존 인덱스 삭제: {VECTOR_INDEX_NAME}")
            collection.drop_search_index(VECTOR_INDEX_NAME)
            _wait_for_index_drop(collection, VECTOR_INDEX_NAME)
        else:
            print(f"  ✅ 이미 존재: {VECTOR_INDEX_NAME}")

    if VECTOR_INDEX_NAME not in existing or force_recreate:
        to_create.append(
            SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": EMBEDDING_FIELD,
                            "numDimensions": EMBEDDING_DIMS,  # voyage-4: 1024차원
                            "similarity": "cosine",
                        }
                    ]
                },
                name=VECTOR_INDEX_NAME,
                type="vectorSearch",
            )
        )

    # ── [2] Atlas Search (전문 검색) 인덱스 ──────────────────
    if SEARCH_INDEX_NAME in existing:
        if force_recreate:
            print(f"  기존 인덱스 삭제: {SEARCH_INDEX_NAME}")
            collection.drop_search_index(SEARCH_INDEX_NAME)
            _wait_for_index_drop(collection, SEARCH_INDEX_NAME)
        else:
            print(f"  ✅ 이미 존재: {SEARCH_INDEX_NAME}")

    if SEARCH_INDEX_NAME not in existing or force_recreate:
        to_create.append(
            SearchIndexModel(
                definition={
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            TEXT_FIELD: {"type": "string"},
                            "title":    {"type": "string"},  # 최상위 레벨 title
                        },
                    }
                },
                name=SEARCH_INDEX_NAME,
                type="search",
            )
        )

    if to_create:
        names = [m.document["name"] for m in to_create]
        print(f"  인덱스 생성 요청: {names}")
        collection.create_search_indexes(to_create)
        return True

    return False


def _wait_for_index_drop(
    collection: pymongo.collection.Collection,
    index_name: str,
    timeout: int = 120,
    poll_interval: int = 3,
) -> None:
    """인덱스 삭제 완료까지 대기 (내부 헬퍼)."""
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
    지정한 인덱스들이 모두 'READY' 상태가 될 때까지 폴링합니다.

    Args:
        collection   : MongoDB 컬렉션
        index_names  : 대기할 인덱스 이름 목록
        timeout      : 최대 대기 시간 (초)
        poll_interval: 폴링 간격 (초)

    Returns:
        모든 인덱스 READY이면 True, 타임아웃이면 False
    """
    target = set(index_names)
    start  = time.time()
    print(f"  인덱스 빌드 대기 중 (최대 {timeout}초)...")

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
            print(f"\n  ✅ 모든 인덱스 READY! ({elapsed}초 소요)")
            return True

        time.sleep(poll_interval)

    print(f"\n  ⚠️  타임아웃 ({timeout}초)")
    return False


def setup_indexes(
    collection: pymongo.collection.Collection,
    force_recreate: bool = False,
    wait_timeout: int = 300,
) -> None:
    """인덱스 생성 + READY 대기를 한 번에 처리하는 편의 함수."""
    print("\n[인덱스 설정]")
    created = create_atlas_indexes(collection, force_recreate=force_recreate)
    if created:
        wait_for_indexes_ready(
            collection,
            index_names=[VECTOR_INDEX_NAME, SEARCH_INDEX_NAME],
            timeout=wait_timeout,
        )
    else:
        print("  → 모든 인덱스가 이미 존재합니다.")


# ============================================================
# 하이브리드 검색: 벡터 검색 + 전문 검색
# ============================================================

def vector_search(
    collection: pymongo.collection.Collection,
    query_embedding: List[float],
    k: int = 10,
) -> List[Dict[str, Any]]:
    """
    $vectorSearch를 사용한 시맨틱(의미 기반) 검색.

    langchain-mongodb는 metadata를 최상위 레벨에 펼쳐서 저장합니다.
    예: {"text": ..., "embedding": [...], "title": ..., "category": ...}
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": EMBEDDING_FIELD,
                "queryVector": query_embedding,
                "numCandidates": k * 10,
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 1,
                TEXT_FIELD: 1,
                "title": 1,       # 최상위 레벨 필드
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
    $search를 사용한 전문 검색(Full-Text Search).
    Atlas Search 인덱스(search_index)가 필요합니다.
    """
    pipeline = [
        {
            "$search": {
                "index": SEARCH_INDEX_NAME,
                "text": {
                    "query": query,
                    "path": [TEXT_FIELD, "title"],  # 최상위 레벨 필드
                },
            }
        },
        {"$limit": k},
        {
            "$project": {
                "_id": 1,
                TEXT_FIELD: 1,
                "title": 1,       # 최상위 레벨 필드
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
    RRF(Reciprocal Rank Fusion)으로 벡터 검색과 전문 검색 결과를 융합합니다.

    RRF 공식:
        RRF_score(d) = Σ  1 / (k + rank_i(d))

    Args:
        vector_results: 벡터 검색 결과
        text_results  : 전문 검색 결과
        rrf_k         : RRF 상수 k (기본값 60)

    Returns:
        RRF 점수 기준 내림차순 정렬 결과
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
    """RRF 결과 테이블을 출력합니다."""
    print("\n" + "=" * 75)
    print("  RRF (Reciprocal Rank Fusion) 검색 결과")
    print("=" * 75)
    print(f"  {'순위':<4} {'문서 제목':<32} {'벡터순위':<8} {'텍스트순위':<10} {'RRF점수':<12} 카테고리")
    print("-" * 75)

    for i, result in enumerate(rrf_results[:top_k], start=1):
        doc      = result["doc"]
        title    = (doc.get("title")    or "제목없음")[:30]
        category = (doc.get("category") or "-")
        v_rank   = str(result["vector_rank"]) if result["vector_rank"] else "-"
        t_rank   = str(result["text_rank"])   if result["text_rank"]   else "-"
        rrf_score = result["rrf_score"]
        print(f"  {i:<4} {title:<32} {v_rank:<8} {t_rank:<10} {rrf_score:.6f}   {category}")

    print("=" * 75)
    print("\n  [점수 상세]")
    for i, result in enumerate(rrf_results[:top_k], start=1):
        doc     = result["doc"]
        title   = (doc.get("title") or "제목없음")[:25]
        v_str   = f"{result['vector_score']:.4f}" if result["vector_score"] is not None else "없음"
        t_str   = f"{result['text_score']:.4f}"   if result["text_score"]   is not None else "없음"
        print(f"  {i}. {title}: 벡터점수={v_str}, 텍스트점수={t_str}, RRF={result['rrf_score']:.6f}")
    print()


def hybrid_search(
    collection: pymongo.collection.Collection,
    query: str,
    embeddings: VoyageAIEmbeddings,
    k: int = 10,
    rrf_k: int = 60,
    verbose: bool = True,
) -> List[Dict]:
    """하이브리드 검색 (벡터 + 전문 검색 + RRF)."""
    query_embedding = embeddings.embed_query(query)
    v_results = vector_search(collection, query_embedding, k=k)
    t_results = text_search(collection, query, k=k)

    if verbose:
        print(f"  → 벡터 검색: {len(v_results)}개  |  전문 검색: {len(t_results)}개")

    rrf_results = reciprocal_rank_fusion(v_results, t_results, rrf_k=rrf_k)

    if verbose:
        print_rrf_results(rrf_results, top_k=5)

    return rrf_results


# ============================================================
# RAG 체인 구성
# ============================================================

def format_context(rrf_results: List[Dict], top_k: int = 3) -> str:
    """RRF 결과 상위 문서로 RAG 컨텍스트를 구성합니다."""
    parts = []
    for i, result in enumerate(rrf_results[:top_k], start=1):
        doc     = result["doc"]
        title   = doc.get("title", "")
        content = doc.get(TEXT_FIELD, "")
        parts.append(
            f"--- 문서 {i}: {title} (RRF 점수: {result['rrf_score']:.4f}) ---\n{content}"
        )
    return "\n\n".join(parts)


def build_rag_chain(llm: ChatOpenAI):
    """LangChain LCEL로 RAG 체인을 구성합니다."""
    prompt = ChatPromptTemplate.from_template(
        """당신은 MongoDB 전문가입니다.
주어진 컨텍스트를 바탕으로 MongoDB 장애와 관련된 질문에 정확하고 실용적인 답변을 제공하세요.

[참고 문서]
{context}

[질문]
{question}

[답변 지침]
1. 참고 문서의 정보를 우선적으로 활용하세요.
2. 구체적인 해결 방법과 명령어를 포함하세요.
3. 단계적으로 설명하세요 (즉각 조치 → 근본 원인 해결 → 예방).

[답변]
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
    """MongoDB 장애 관련 질문에 RAG 기반으로 답변합니다."""
    print(f"\n{'='*75}")
    print(f"  질문: {question}")
    print(f"{'='*75}")

    print("\n[검색 중...]")
    rrf_results = hybrid_search(
        collection=collection, query=question, embeddings=embeddings,
        k=search_k, rrf_k=rrf_k, verbose=verbose,
    )

    context = format_context(rrf_results, top_k=context_top_k)

    print("[답변 생성 중...]")
    answer = build_rag_chain(llm).invoke({"context": context, "question": question})

    print(f"\n{'─'*75}")
    print("  [답변]")
    print(f"{'─'*75}")
    print(answer)
    print(f"{'='*75}\n")
    return answer


# ============================================================
# 메인 실행
# ============================================================

def main():
    """메인 실행 함수"""

    # ── 1. 컴포넌트 초기화 ──────────────────────────────────────
    print("\n" + "="*50)
    print(" [1/5] 컴포넌트 초기화")
    print("="*50)
    client     = get_mongodb_client()
    embeddings = get_embeddings()
    llm        = get_llm()
    collection = client[DB_NAME][COLLECTION_NAME]

    # ── 2. 문서 로드 ─────────────────────────────────────────────
    print("\n" + "="*50)
    print(" [2/5] 지식 베이스 문서 로드")
    print("="*50)
    docs = prepare_documents()
    load_documents_to_atlas(collection, embeddings, docs, force_reload=False)

    # ── 3. 인덱스 자동 생성 ───────────────────────────────────────
    print("\n" + "="*50)
    print(" [3/5] Atlas 인덱스 자동 생성")
    print("="*50)
    setup_indexes(collection, force_recreate=False, wait_timeout=300)

    # ── 4. 예시 질문 실행 ─────────────────────────────────────────
    # print("\n" + "="*50)
    # print(" [4/5] 예시 질문 실행")
    # print("="*50)
    # sample_questions = [
    #     "MongoDB connection pool이 고갈되었을 때 어떻게 해결하나요?",
    #     "Replication lag이 발생했을 때 원인과 해결 방법은?",
    #     "쿼리가 너무 느릴 때 어떻게 최적화하나요?",
    # ]
    # for question in sample_questions:
    #     ask_mongodb_question(
    #         question=question, collection=collection,
    #         embeddings=embeddings, llm=llm, context_top_k=3, verbose=True,
    #     )
    #     time.sleep(1)

    # ── 5. 대화형 모드 ─────────────────────────────────────────────
    print("\n" + "="*50)
    print(" [5/5] 대화형 Q&A 모드")
    print("="*50)
    print("MongoDB 장애 관련 질문을 입력하세요. (종료: q)\n")
    print(" 질문 예시 ")
    print(" - MongoDB connection pool이 고갈되었을 때 어떻게 해결하나요? ")
    print(" - Replication lag이 발생했을 때 원인과 해결 방법은?")
    print(" - 쿼리가 너무 느릴 때 어떻게 최적화하나요? ")
    while True:
        try:
            question = input("질문 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n프로그램을 종료합니다.")
            break
        if question.lower() in {"q", "quit", "exit", "종료"}:
            print("프로그램을 종료합니다.")
            break
        if question:
            ask_mongodb_question(
                question=question, collection=collection,
                embeddings=embeddings, llm=llm, context_top_k=3, verbose=True,
            )

    client.close()


if __name__ == "__main__":
    main()
