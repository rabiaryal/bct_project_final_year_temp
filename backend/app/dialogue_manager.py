"""
Dialogue Manager — 10-Intent Aggregation Pipeline

Pipeline:
  1. NLU (BERT Intent + RoBERTa Entities)
  2. Keyword Override (low-confidence correction)
  3. Entity Fix (LOCATION -> COLLEGE_NAME reclassification)
  4. Slot Manager (entity -> slot, normalize, context update)
  5. Template Check (actionable?)
     a. Static intent (greeting/goodbye) -> respond immediately
     b. Missing required slots -> follow-up question
     c. Actionable -> build pipeline -> aggregate -> format response
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import random
import uuid
import logging
import re

from app.nlu import BERTIntentClassifier, RoBERTaEntityExtractor
from app.context.slot_manager import SlotManager, DialogueContext
from app.templates.intent_templates import get_template, INTENT_TEMPLATES
from app.handlers.router import route_intent
from app.utils.formatter import format_response
from app.repositories.mongo_client import MongoRepository
from app.schemas import ChatRequest, ChatResponse
from app.utils.config import config

logger = logging.getLogger(__name__)


# ============================================================================
# KEYWORD -> INTENT OVERRIDE RULES
# ============================================================================

_KEYWORD_OVERRIDES = [
    (re.compile(r"\b(recommend|suggest|help me choose|best college|which college should)\b", re.I),
     "personalized_recommendation", 0.82),
    (re.compile(r"\b(best|top|highest rated|top rated)\b", re.I),
     "best_items_search", 0.78),
    (re.compile(r"\b(compare|vs|versus|between|difference)\b", re.I),
     "compare_colleges", 0.78),
    (re.compile(r"\b(hostel|accommodation|dorm(?:itory)?|boarding|residential)\b", re.I),
     "hostel_query", 0.78),
    (re.compile(r"\b(contact|phone number|phone no|call|email|reach|helpline)\b", re.I),
     "contact_query", 0.78),
    (re.compile(r"\b(detail|details|info|information|about|tell me about)\b.*\b(college|campus|institute)\b", re.I),
     "college_details", 0.72),
    (re.compile(r"\b(find|search|show|list|colleges? in|which colleges?)\b", re.I),
     "search_college", 0.72),
]

# College-name indicator words
_COLLEGE_NAME_INDICATORS = re.compile(
    r"\b(campus|college|institute|engineering|university|management|polytechnic)\b", re.I
)


# ============================================================================
# TERMINAL COLORS
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def _print_header(session_id: str, message: str):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}", flush=True)
    print(f"{Colors.BOLD}  PIPELINE - Session: {session_id[:12]}...{Colors.END}", flush=True)
    print(f"{Colors.BOLD}  Message: {Colors.END}{message}", flush=True)
    print(f"{Colors.HEADER}{'='*70}{Colors.END}", flush=True)


def _print_nlu(intent: str, confidence: float, metadata: Dict, entities: List[Dict], extracted: Dict):
    print(f"\n{Colors.CYAN}--- NLU ---{Colors.END}", flush=True)
    print(f"  Intent: {Colors.GREEN}{intent}{Colors.END} ({confidence:.2%})", flush=True)
    if metadata and 'top_predictions' in metadata:
        for name, prob in metadata['top_predictions'][:3]:
            print(f"    {name}: {prob:.2%}", flush=True)
    print(f"  Entities: {extracted if extracted else '(none)'}", flush=True)


def _print_slots(prev: Dict, ctx: Any):
    print(f"\n{Colors.YELLOW}--- SLOTS ---{Colors.END}", flush=True)
    if ctx.slots:
        for k, v in ctx.slots.items():
            marker = "(new)" if k not in prev else ("(upd)" if prev.get(k) != v else "")
            print(f"  {k}: {v} {marker}", flush=True)
    else:
        print("  (empty)", flush=True)
    print(f"  Missing: {ctx.missing_slots or '(none)'}", flush=True)
    print(f"  Actionable: {'Yes' if ctx.is_actionable else 'No'}", flush=True)


def _print_context(ctx: Any):
    print(f"\n{Colors.BLUE}--- CONTEXT ---{Colors.END}", flush=True)
    print(f"  Turn: {ctx.turn_count}", flush=True)
    print(f"  Intent: {ctx.current_intent} (prev: {ctx.previous_intent or 'none'})", flush=True)
    print(f"  Family: {ctx.intent_family}", flush=True)
    if ctx.slots:
        print(f"  Accumulated slots:", flush=True)
        for k, v in ctx.slots.items():
            print(f"    {k}: {v}", flush=True)
    else:
        print(f"  Accumulated slots: (empty)", flush=True)


def _print_result(response_type: str, count: int = 0):
    print(f"\n{Colors.GREEN}--- RESPONSE ({response_type}) ---{Colors.END}", flush=True)
    if count:
        print(f"  Results: {count}", flush=True)


def _print_final(response: str, elapsed: float):
    print(f"\n{Colors.BOLD}BOT:{Colors.END}", flush=True)
    # Print just first 200 chars to avoid flooding terminal
    preview = response[:200] + ("..." if len(response) > 200 else "")
    print(f"  {preview}", flush=True)
    print(f"  ({elapsed:.3f}s)\n", flush=True)


# ============================================================================
# DIALOGUE MANAGER
# ============================================================================

class DialogueManager:

    def __init__(self):
        self.intent_classifier: Optional[BERTIntentClassifier] = None
        self.entity_extractor: Optional[RoBERTaEntityExtractor] = None
        self.slot_manager = SlotManager()
        self.mongo_repo: Optional[MongoRepository] = None
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return
        logger.info("Initializing Dialogue Manager...")

        self.intent_classifier = BERTIntentClassifier()
        self.entity_extractor = RoBERTaEntityExtractor()

        self.mongo_repo = MongoRepository()
        await self.mongo_repo.connect()

        self._initialized = True
        logger.info(f"Dialogue Manager ready ({len(INTENT_TEMPLATES)} intents)")

    async def shutdown(self):
        if self.mongo_repo:
            await self.mongo_repo.disconnect()
        self.slot_manager.contexts.clear()
        logger.info("Dialogue Manager shutdown")

    # ------------------------------------------------------------------
    # MAIN PIPELINE
    # ------------------------------------------------------------------

    async def process_message(self, request: ChatRequest) -> ChatResponse:
        start_time = datetime.now()
        session_id = request.session_id or self._generate_session_id()

        try:
            _print_header(session_id, request.message)

            # ── Step 1: NLU ──────────────────────────────────────────────
            intent, confidence, metadata = await self.intent_classifier.predict(request.message)
            entity_list, _ner_meta = await self.entity_extractor.predict(request.message)
            entities = self._entities_to_dict(entity_list, threshold=0.1)

            # Keyword correction + entity fix
            intent, confidence = self._apply_keyword_corrections(request.message, intent, confidence)
            entities = self._fix_college_name_entity(entities, request.message)

            # ── Fix #1: Intent lock during slot filling ──────────────────
            # If the previous turn asked a follow-up question (missing required
            # slots), lock to that intent so a bare reply like "computer"
            # doesn't get reclassified as greeting/unknown/etc.
            prev_context = self.slot_manager.get_context(session_id)
            if prev_context and prev_context.pending_slot:
                locked_intent = prev_context.current_intent
                if intent != locked_intent:
                    logger.info(
                        f"Intent lock: {intent} -> {locked_intent} "
                        f"(pending slot: {prev_context.pending_slot})"
                    )
                    intent = locked_intent

            # ── Fix #7: Low-confidence fallback ──────────────────────────
            # If NLU is very uncertain, keep the previous intent instead
            # of switching to a likely-wrong prediction.
            elif (
                prev_context
                and prev_context.current_intent
                and prev_context.current_intent not in ("greeting", "goodbye", "unknown")
                and confidence < 0.45
            ):
                logger.info(
                    f"Low-confidence fallback: {intent} ({confidence:.0%}) "
                    f"-> keeping {prev_context.current_intent}"
                )
                intent = prev_context.current_intent

            _print_nlu(intent, confidence, metadata, entity_list, entities)
            logger.info(f"[{session_id[:8]}] Intent: {intent} ({confidence:.2f}) | Entities: {entities}")

            # ── Step 2: Slot Manager ─────────────────────────────────────
            prev_slots = dict(prev_context.slots) if prev_context else {}

            context = self.slot_manager.process_turn(
                session_id, intent, entities, raw_text=request.message
            )

            _print_slots(prev_slots, context)
            _print_context(context)

            # ── Step 3: Template ─────────────────────────────────────────
            template = get_template(intent)

            # ── Static intents (no DB) ───────────────────────────────────
            # Fix #3: Greeting mid-conversation should not wipe state.
            # We respond with greeting text but preserve slots.
            if intent in ("greeting", "goodbye", "unknown"):
                response_msg = format_response(intent, [], context.slots)
                _print_result("STATIC", 0)
                elapsed = (datetime.now() - start_time).total_seconds()
                _print_final(response_msg, elapsed)
                return self._build_response(
                    session_id, response_msg, intent, entities,
                    confidence, start_time,
                    {"slots": dict(context.slots), "type": "static",
                     "context": context.to_dict()},
                )

            # ── Missing required slots → follow-up ───────────────────────
            # (hybrid handlers also do their own slot check, but this
            #  catches the standard intents that use templates)
            if not context.is_actionable and intent not in (
                "recommend_with_constraints", "personalized_recommendation"
            ):
                follow_up = self.slot_manager.get_follow_up(context)
                # pending_slot is set inside get_follow_up()
                _print_result("FOLLOW-UP", 0)
                elapsed = (datetime.now() - start_time).total_seconds()
                _print_final(follow_up, elapsed)
                return self._build_response(
                    session_id, follow_up, intent, entities,
                    confidence, start_time,
                    {"slots": dict(context.slots), "missing": context.missing_slots,
                     "type": "follow_up", "context": context.to_dict()},
                )

            # ── Route to handler ─────────────────────────────────────────
            result = await route_intent(
                intent=intent,
                slots=context.slots,
                collection=self.mongo_repo.collection,
                top_k=5,
            )

            # Hybrid handlers may return a follow-up question
            if result.get("action") == "ask":
                # Record which slot the bot is waiting for
                if result.get("missing_slot"):
                    context.pending_slot = result["missing_slot"]
                    logger.info(f"Hybrid pending_slot = {result['missing_slot']}")
                _print_result("FOLLOW-UP (hybrid)", 0)
                elapsed = (datetime.now() - start_time).total_seconds()
                _print_final(result["response"], elapsed)
                return self._build_response(
                    session_id, result["response"], intent, entities,
                    confidence, start_time,
                    {"slots": dict(context.slots), "missing_slot": result.get("missing_slot"),
                     "type": "follow_up", "context": context.to_dict()},
                )

            # Persist results back into context
            self.slot_manager.update_with_results(
                session_id, result["results"], result.get("query", {})
            )

            _print_result("DB RETRIEVAL", result["count"])
            elapsed = (datetime.now() - start_time).total_seconds()
            _print_final(result["response"], elapsed)

            return self._build_response(
                session_id, result["response"], intent, entities,
                confidence, start_time,
                {
                    "slots": dict(context.slots),
                    "query": result.get("query"),
                    "count": result["count"],
                    "type": "retrieval",
                    "context": context.to_dict(),
                },
            )

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return self._build_response(
                session_id,
                "I'm sorry, something went wrong. Please try again.",
                "error", {}, 0.0, start_time,
                {"error": str(e)},
            )

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _apply_keyword_corrections(self, message: str, intent: str, confidence: float):
        for pattern, target, max_conf in _KEYWORD_OVERRIDES:
            if confidence < max_conf and pattern.search(message):
                if target != intent:
                    logger.info(f"Keyword override: {intent} -> {target} (conf={confidence:.0%})")
                    return target, confidence
        return intent, confidence

    def _fix_college_name_entity(self, entities: Dict[str, str], message: str) -> Dict[str, str]:
        """Reclassify LOCATION -> COLLEGE_NAME when context indicates a college."""
        if "LOCATION" not in entities or "COLLEGE_NAME" in entities:
            return entities

        loc_value = entities["LOCATION"]

        # Value itself contains indicator
        if _COLLEGE_NAME_INDICATORS.search(loc_value):
            entities["COLLEGE_NAME"] = entities.pop("LOCATION")
            return entities

        # Location followed by indicator in message (e.g. "Pulchowk Engineering Campus")
        escaped = re.escape(loc_value)
        nearby = re.search(
            escaped + r"[\s,]+" + r"(?:campus|college|institute|engineering|university|management)",
            message, re.I,
        )
        if nearby:
            entities["COLLEGE_NAME"] = nearby.group(0).strip()
            entities.pop("LOCATION")

        return entities

    def _entities_to_dict(self, entity_list: list, threshold: float = 0.1) -> Dict[str, str]:
        entities: Dict[str, str] = {}
        for ent in entity_list:
            conf = ent.get("confidence", 1.0)
            if conf < threshold:
                continue
            etype = ent.get("type", "").upper()
            evalue = ent.get("value", ent.get("text", ""))
            if etype and evalue and etype not in entities:
                entities[etype] = evalue
        return entities

    def _generate_session_id(self) -> str:
        return f"session_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"

    def _build_response(
        self, session_id, message, intent, entities,
        confidence, start_time, debug_info=None,
    ) -> ChatResponse:
        elapsed = (datetime.now() - start_time).total_seconds()
        return ChatResponse(
            message=message,
            session_id=session_id,
            intent=intent,
            entities=entities,
            confidence=confidence,
            timestamp=datetime.now(),
            debug_info={**(debug_info or {}), "processing_time": elapsed},
        )

    def get_session_debug(self, session_id: str) -> Dict[str, Any]:
        return self.slot_manager.get_debug_info(session_id)

    async def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "components": {
                "intent_classifier": {"status": "loaded" if self.intent_classifier else "not_loaded"},
                "entity_extractor": {"status": "loaded" if self.entity_extractor else "not_loaded"},
                "mongodb": {"status": "connected" if (self.mongo_repo and self.mongo_repo.collection is not None) else "not_connected"},
            },
            "active_sessions": len(self.slot_manager.contexts),
        }


# Singleton
dialogue_manager = DialogueManager()
