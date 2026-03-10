"""
Scorer — Python-side scoring logic for hybrid handlers.

Two scorers with different weight profiles:
  - RecommendScorer   (recommend_with_constraints)
  - PersonalScorer    (personalized_recommendation)
"""

from typing import Dict, Any, List, Tuple, Optional


# ============================================================================
# ADMISSION SAFETY (personal handler only)
# ============================================================================

_SAFETY_THRESHOLDS = {
    "SAFE":     500,
    "MODERATE": 200,
    "RISKY":    0,
}


def get_admission_safety(rank_gap: int) -> Tuple[str, str, str]:
    """
    Classify admission safety from rank gap (cutoff − user_rank).

    Returns (label, emoji, advice).
    """
    if rank_gap >= _SAFETY_THRESHOLDS["SAFE"]:
        return "SAFE", "🟢", "High chance of admission"
    elif rank_gap >= _SAFETY_THRESHOLDS["MODERATE"]:
        return "MODERATE", "🟡", "Good chance — apply with confidence"
    else:
        return "RISKY", "🔴", "Low chance — consider as backup only"


# ============================================================================
# RECOMMEND SCORER  (rank=35, fee=35, rating=20, hostel=5, public=5)
# ============================================================================

class RecommendScorer:
    """Score + rerank for recommend_with_constraints intent."""

    WEIGHTS = {
        "rank_safety":  35,
        "fee_saving":   35,
        "rating":       20,
        "hostel_bonus":  5,
        "public_bonus":  5,
    }

    @classmethod
    def score(cls, college: Dict[str, Any], slots: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Score a single candidate against user constraints.

        Returns (score 0–100, list of explanation strings).
        """
        score = 0.0
        reasons: List[str] = []

        user_rank   = int(slots.get("rank", 0) or 0)
        user_budget = int(slots.get("budget", 0) or 0)
        cutoff      = college.get("cutoff_rank", 0) or 0
        fee         = college.get("fee", 0) or 0
        rating      = college.get("rating", 0) or 0

        # ── 1. Rank safety (0–35) ──────────────────
        if user_rank and cutoff:
            gap = cutoff - user_rank
            if gap <= 0:
                rank_score = 5
                reasons.append(
                    f"⚠️  Rank {user_rank} exactly meets cutoff (very competitive)"
                )
            elif gap <= 200:
                rank_score = 12
                reasons.append(
                    f"Rank {user_rank} is close to cutoff (safety gap: {gap} ranks)"
                )
            elif gap <= 500:
                rank_score = 20
                reasons.append(
                    f"Rank {user_rank} safely within cutoff (gap: {gap} ranks)"
                )
            elif gap <= 1000:
                rank_score = 28
                reasons.append(
                    f"Rank {user_rank} comfortably within cutoff (gap: {gap} ranks)"
                )
            else:
                rank_score = 35
                reasons.append(
                    f"Rank {user_rank} very safely qualifies (gap: {gap} ranks)"
                )
            score += rank_score
        else:
            score += cls.WEIGHTS["rank_safety"] / 2

        # ── 2. Fee saving (0–35) ───────────────────
        if user_budget and fee:
            saving = user_budget - fee
            saving_pct = saving / user_budget if user_budget else 0

            if saving_pct >= 0.80:
                fee_score = 35
                reasons.append(
                    f"Very affordable: Rs.{fee:,.0f} fee "
                    f"(saves Rs.{saving:,.0f} = {saving_pct*100:.0f}% of budget)"
                )
            elif saving_pct >= 0.50:
                fee_score = 26
                reasons.append(
                    f"Affordable: Rs.{fee:,.0f} fee "
                    f"(saves Rs.{saving:,.0f} = {saving_pct*100:.0f}% of budget)"
                )
            elif saving_pct >= 0.20:
                fee_score = 17
                reasons.append(f"Reasonable: Rs.{fee:,.0f} fee (saves Rs.{saving:,.0f})")
            else:
                fee_score = 8
                reasons.append(f"Close to budget: Rs.{fee:,.0f} fee (saves Rs.{saving:,.0f})")
            score += fee_score
        else:
            score += cls.WEIGHTS["fee_saving"] / 2

        # ── 3. Rating (0–20) ──────────────────────
        if rating:
            score += (rating / 5.0) * cls.WEIGHTS["rating"]
            if rating >= 4.8:
                reasons.append(f"Exceptional rating: {rating}/5.0")
            elif rating >= 4.5:
                reasons.append(f"Excellent rating: {rating}/5.0")
            elif rating >= 4.0:
                reasons.append(f"Good rating: {rating}/5.0")
            else:
                reasons.append(f"Average rating: {rating}/5.0")

        # ── 4. Hostel bonus (0 or 5) ──────────────
        if college.get("hostel"):
            score += cls.WEIGHTS["hostel_bonus"]
            reasons.append("Hostel available on campus")

        # ── 5. Public bonus (0 or 5) ──────────────
        ctype = str(college.get("college_type", "")).upper()
        if ctype == "PUBLIC":
            score += cls.WEIGHTS["public_bonus"]
            reasons.append("Government college — reputed and affordable")

        return round(score, 2), reasons

    @classmethod
    def rerank(
        cls,
        candidates: List[Dict[str, Any]],
        slots: Dict[str, Any],
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        scored = []
        for c in candidates:
            s, reasons = cls.score(c, slots)
            scored.append({**c, "Score": s, "Reasons": reasons})
        scored.sort(key=lambda x: (x["Score"], x.get("rating", 0)), reverse=True)
        return scored[:top_n]


# ============================================================================
# PERSONAL SCORER  (rank=45, fee=25, rating=20, hostel=5, public=5)
# ============================================================================

class PersonalScorer:
    """Score + rerank for personalized_recommendation intent."""

    WEIGHTS = {
        "rank_safety":  45,
        "fee_saving":   25,
        "rating":       20,
        "hostel_bonus":  5,
        "public_bonus":  5,
    }

    @classmethod
    def score(
        cls, college: Dict[str, Any], slots: Dict[str, Any],
    ) -> Tuple[float, List[str], str]:
        """
        Score a single candidate for personalized recommendation.

        Returns (score 0–100, reasons, safety_label).
        """
        score = 0.0
        reasons: List[str] = []

        user_rank   = int(slots.get("rank", 0) or 0)
        user_budget = int(slots.get("budget", 0) or 0)
        cutoff      = college.get("cutoff_rank", 0) or 0
        fee         = college.get("fee", 0) or 0
        rating      = college.get("rating", 0) or 0

        # ── 1. Rank safety (0–45) ──────────────────
        safety_label = "UNKNOWN"
        if user_rank and cutoff:
            gap = cutoff - user_rank
            safety_label, safety_emoji, safety_msg = get_admission_safety(gap)

            if gap <= 0:
                rank_score = 5
                reasons.append(
                    f"{safety_emoji} Admission: {safety_label} — "
                    f"cutoff exactly matches your rank {user_rank}"
                )
            elif gap <= 200:
                rank_score = 15
                reasons.append(
                    f"{safety_emoji} Admission: {safety_label} — "
                    f"{safety_msg} (rank gap: {gap})"
                )
            elif gap <= 500:
                rank_score = 28
                reasons.append(
                    f"{safety_emoji} Admission: {safety_label} — "
                    f"{safety_msg} (rank gap: {gap})"
                )
            elif gap <= 1000:
                rank_score = 37
                reasons.append(
                    f"{safety_emoji} Admission: {safety_label} — "
                    f"{safety_msg} (rank gap: {gap})"
                )
            else:
                rank_score = 45
                reasons.append(
                    f"{safety_emoji} Admission: {safety_label} — "
                    f"{safety_msg} (rank gap: {gap})"
                )
            score += rank_score
        else:
            score += cls.WEIGHTS["rank_safety"] / 2

        # ── 2. Fee saving (0–25) ───────────────────
        if user_budget and fee:
            saving = user_budget - fee
            saving_pct = saving / user_budget if user_budget else 0

            if saving_pct >= 0.80:
                fee_score = 25
                reasons.append(
                    f"Well within your budget (Rs.{fee:,.0f} vs your Rs.{user_budget:,.0f})"
                )
            elif saving_pct >= 0.50:
                fee_score = 18
                reasons.append(
                    f"Affordable for you (Rs.{fee:,.0f} fee, saves Rs.{saving:,.0f})"
                )
            elif saving_pct >= 0.20:
                fee_score = 11
                reasons.append(f"Within your budget (Rs.{fee:,.0f} fee)")
            else:
                fee_score = 5
                reasons.append(
                    f"Tight on budget (Rs.{fee:,.0f} fee, only Rs.{saving:,.0f} to spare)"
                )
            score += fee_score
        else:
            score += cls.WEIGHTS["fee_saving"] / 2

        # ── 3. Rating (0–20) ──────────────────────
        if rating:
            score += (rating / 5.0) * cls.WEIGHTS["rating"]
            if rating >= 4.8:
                reasons.append(f"Exceptional college — rated {rating}/5.0")
            elif rating >= 4.5:
                reasons.append(f"Excellent college — rated {rating}/5.0")
            elif rating >= 4.0:
                reasons.append(f"Good college — rated {rating}/5.0")
            else:
                reasons.append(f"Average rating — {rating}/5.0")

        # ── 4. Hostel bonus (0 or 5) ──────────────
        if college.get("hostel"):
            score += cls.WEIGHTS["hostel_bonus"]
            reasons.append("Hostel available — good if you are from outside")

        # ── 5. Public bonus (0 or 5) ──────────────
        ctype = str(college.get("college_type", "")).upper()
        if ctype == "PUBLIC":
            score += cls.WEIGHTS["public_bonus"]
            reasons.append("Government college — strong industry reputation in Nepal")

        return round(score, 2), reasons, safety_label

    @classmethod
    def rerank(
        cls,
        candidates: List[Dict[str, Any]],
        slots: Dict[str, Any],
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        scored = []
        for c in candidates:
            s, reasons, safety = cls.score(c, slots)
            scored.append({
                **c,
                "Score": s,
                "Reasons": reasons,
                "SafetyLabel": safety,
            })
        scored.sort(key=lambda x: (x["Score"], x.get("rating", 0)), reverse=True)
        return scored[:top_n]
