"""
Chat-Endpunkte — OpenAI-kompatible API für Open WebUI
=====================================================

Implementiert /v1/chat/completions und /v1/models.
Integriert RAG-Pipeline: Frage → Embedding → Qdrant → Kontext → LLM.
"""

import json
import time
import uuid
from typing import Optional

import httpx
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from app.config import get_settings

router = APIRouter()

# ---------------------------------------------------------------------------
# Psychiatrischer System-Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Du bist ein hochqualifizierter psychiatrischer KI-Assistent für klinische Fachkräfte.

REGELN:
1. Antworte AUSSCHLIESSLICH auf Basis der bereitgestellten Kontextinformationen.
2. Wenn die bereitgestellten Informationen nicht ausreichen, sage klar:
   "Die verfügbaren Unterlagen enthalten keine ausreichende Evidenz zu dieser Frage."
3. Halluziniere NIEMALS Diagnosen, Medikamente, Dosierungen oder Behandlungsempfehlungen.
4. Zitiere die Quelle (Dokumentname, Seite) bei jeder faktischen Aussage.
5. Verwende korrekte psychiatrische Fachterminologie (ICD-11, DSM-5-TR).
6. Kennzeichne deine Reasoning-Schritte klar und nachvollziehbar.
7. Bei Unsicherheit gib den Konfidenzgrad an (hoch/mittel/niedrig).
8. Weise bei kritischen Entscheidungen immer darauf hin, dass eine ärztliche Überprüfung erforderlich ist.

KONTEXT AUS WISSENSDATENBANK:
{context}
"""


# ---------------------------------------------------------------------------
# Request/Response Modelle (OpenAI-kompatibel)
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


# ---------------------------------------------------------------------------
# /v1/models — Modell-Liste für Open WebUI
# ---------------------------------------------------------------------------
@router.get("/models")
async def list_models():
    settings = get_settings()
    model_id = settings.llm_model.split("/")[-1]
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "permission": [],
            }
        ],
    }


# ---------------------------------------------------------------------------
# /v1/chat/completions — Haupt-Chat-Endpunkt mit RAG
# ---------------------------------------------------------------------------
@router.post("/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    settings = get_settings()

    # Letzte User-Nachricht extrahieren
    user_message = ""
    for msg in reversed(body.messages):
        if msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        raise HTTPException(status_code=400, detail="Keine User-Nachricht gefunden.")

    # --- RAG: Relevante Kontexte aus Qdrant abrufen ---
    context = ""
    sources = []
    try:
        vectorstore = request.app.state.vectorstore
        results = await vectorstore.search(user_message, top_k=settings.rag_top_k)

        if results:
            context_parts = []
            for i, result in enumerate(results, 1):
                score = result.get("score", 0)
                if score >= settings.rag_similarity_threshold:
                    text = result.get("text", "")
                    source = result.get("metadata", {}).get("source", "Unbekannt")
                    page = result.get("metadata", {}).get("page", "?")
                    context_parts.append(
                        f"[Quelle {i}: {source}, S.{page}] (Relevanz: {score:.2f})\n{text}"
                    )
                    sources.append({"source": source, "page": page, "score": score})

            context = "\n\n---\n\n".join(context_parts)
            logger.info(
                f"RAG: {len(context_parts)} relevante Kontexte gefunden "
                f"(von {len(results)} Ergebnissen)"
            )
    except Exception as e:
        logger.warning(f"RAG-Suche fehlgeschlagen: {e}")
        context = "[Wissensdatenbank nicht verfügbar]"

    # --- System-Prompt mit Kontext zusammenbauen ---
    system_content = SYSTEM_PROMPT.format(
        context=context if context else "[Keine relevanten Dokumente gefunden]"
    )

    # Nachrichten für LLM vorbereiten
    llm_messages = [{"role": "system", "content": system_content}]
    for msg in body.messages:
        if msg.role != "system":
            llm_messages.append({"role": msg.role, "content": msg.content})

    # --- Anfrage an vLLM weiterleiten ---
    llm_payload = {
        "model": settings.llm_model,
        "messages": llm_messages,
        "temperature": body.temperature or settings.llm_temperature,
        "max_tokens": body.max_tokens or settings.llm_max_tokens,
        "stream": body.stream or False,
    }
    if body.top_p is not None:
        llm_payload["top_p"] = body.top_p

    try:
        if body.stream:
            return StreamingResponse(
                _stream_llm_response(settings.llm_base_url, llm_payload),
                media_type="text/event-stream",
            )
        else:
            return await _complete_llm_response(
                settings.llm_base_url, llm_payload, sources
            )
    except httpx.HTTPError as e:
        logger.error(f"LLM-Anfrage fehlgeschlagen: {e}")
        raise HTTPException(status_code=502, detail=f"LLM nicht erreichbar: {e}")


async def _stream_llm_response(base_url: str, payload: dict):
    """Streamt die LLM-Antwort als SSE-Events."""
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield f"{line}\n\n"
                elif line.strip() == "":
                    continue
    yield "data: [DONE]\n\n"


async def _complete_llm_response(
    base_url: str, payload: dict, sources: list
) -> dict:
    """Nicht-Streaming LLM-Antwort mit Quellenangaben."""
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        result = resp.json()

    # Quellenangaben an Antwort anhängen
    if sources and result.get("choices"):
        source_text = "\n\n---\n**Quellen:**\n"
        for s in sources:
            source_text += f"- {s['source']}, Seite {s['page']} (Relevanz: {s['score']:.0%})\n"
        result["choices"][0]["message"]["content"] += source_text

    return result
