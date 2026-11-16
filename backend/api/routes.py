
import logging
import time
from typing import List

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)

from backend.api.auth import get_current_user

from backend.models.schemas import (
    AnalysisResponse,
    ComponentScores,
    JDComparison,
    SkillValidationDetails,
)

logger = logging.getLogger("ats_resume_scorer")

router = APIRouter(
    prefix="/api/v1",
    tags=["Analysis"],
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def _clean(text: str) -> str:
    for prefix in ("âœ…", "ðŸŒŸ", "â", "ðŸ“"):
        text = text.lstrip(prefix)

    return text.strip()


# =========================================================
# RESUME ANALYSIS
# =========================================================

@router.post(
    "/analyze-resume",
    response_model=AnalysisResponse
)
async def analyze_resume(
    request: Request,
    resume: UploadFile = File(
        ...,
        description="Resume file - PDF or DOCX, max 5 MB"
    ),
    job_description: str = Form(
        "",
        description="Job description text (optional)"
    ),
    user_id: str = Depends(get_current_user),
):

    total_start = time.perf_counter()

    filename = resume.filename or "resume"

    logger.info(
        "[ANALYSIS] Starting analysis for %s",
        filename
    )

    warnings: List[str] = []

    # -----------------------------------------------------
    # LOAD PRELOADED MODELS
    # -----------------------------------------------------

    nlp = request.app.state.nlp
    embedder = request.app.state.embedder

    # -----------------------------------------------------
    # STEP 1: READ UPLOADED FILE
    # -----------------------------------------------------

    try:

        read_start = time.perf_counter()

        file_bytes = await resume.read()

        logger.info(
            "[TIMING] File reading: %.2f seconds",
            time.perf_counter() - read_start
        )

        from backend.services.resume_parser import (
            parse_resume_file,
        )

        # -------------------------------------------------
        # STEP 2: EXTRACT RESUME TEXT
        # -------------------------------------------------

        extraction_start = time.perf_counter()

        resume_text, metadata = parse_resume_file(
            file_bytes,
            filename
        )

        logger.info(
            "[TIMING] File extraction: %.2f seconds",
            time.perf_counter() - extraction_start
        )

        logger.info(
            "[ANALYSIS] Extracted %s characters from %s",
            len(resume_text),
            filename
        )

    except Exception as exc:

        logger.exception(
            "[ERROR] File parsing failed"
        )

        raise HTTPException(
            status_code=422,
            detail=f"Could not read or parse the resume: {exc}",
        )

    # -----------------------------------------------------
    # STEP 3: FULL RESUME ANALYSIS
    # -----------------------------------------------------

    try:

        from backend.services.resume_analyzer import (
            analyze_full_resume,
        )

        analysis_start = time.perf_counter()

        logger.info(
            "[ANALYSIS] Starting full analysis pipeline"
        )

        result = analyze_full_resume(
            resume_text=resume_text,
            nlp=nlp,
            embedder=embedder,
            job_description=job_description,
        )

        logger.info(
            "[TIMING] Full analysis pipeline: %.2f seconds",
            time.perf_counter() - analysis_start
        )

    except Exception as exc:

        logger.exception(
            "[ERROR] Full analysis pipeline failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline failed: {exc}"
        )

    # -----------------------------------------------------
    # STEP 4: JOB DESCRIPTION COMPARISON
    # -----------------------------------------------------

    jd_comparison_result = None

    try:

        if result.get("jd_comparison"):

            jd_data = result["jd_comparison"]

            jd_comparison_result = JDComparison(
                match_percentage=round(
                    float(
                        jd_data.get(
                            "match_percentage", 0.0
                        )
                    ),
                    1
                ),

                semantic_similarity=round(
                    float(
                        jd_data.get(
                            "semantic_similarity", 0.0
                        )
                    ),
                    3
                ),

                matched_keywords=jd_data.get(
                    "matched_keywords", []
                )[:20],

                missing_keywords=jd_data.get(
                    "missing_keywords", []
                )[:15],

                skills_gap=jd_data.get(
                    "skills_gap", []
                )[:10],
            )

    except Exception as exc:

        logger.exception(
            "[ERROR] JD comparison response conversion failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"JD comparison conversion failed: {exc}"
        )

    # -----------------------------------------------------
    # STEP 5: SKILL VALIDATION DETAILS
    # -----------------------------------------------------

    detailed_fb = result.get(
        "detailed_feedback", []
    )

    svd_raw = result.get(
        "skill_validation_details"
    ) or {}

    skill_val_details = SkillValidationDetails(
        validated=svd_raw.get("validated", []),

        unvalidated=svd_raw.get("unvalidated", []),

        total=svd_raw.get("total", 0),

        validated_count=svd_raw.get(
            "validated_count", 0
        ),

        validation_pct=svd_raw.get(
            "validation_pct", 0.0
        ),
    )

    # -----------------------------------------------------
    # STEP 6: BUILD API RESPONSE
    # -----------------------------------------------------

    try:

        response = AnalysisResponse(

            ATS_score=result["ats_score"],

            component_scores=ComponentScores(
                **result["component_scores"]
            ),

            issues_summary=result["issues_summary"],

            detailed_feedback=detailed_fb,

            jd_match_analysis=jd_comparison_result,

            skill_validation_details=skill_val_details,

            # Backward compatibility fields

            ats_score=result["ats_score"],

            keyword_match=(
                jd_comparison_result.match_percentage
                if jd_comparison_result
                else 0.0
            ),

            missing_keywords=result.get(
                "missing_keywords", []
            ),

            matched_keywords=result.get(
                "matched_keywords", []
            ),

            skills=list(
                result.get("skills", [])[:20]
            ),

            jd_comparison=jd_comparison_result,

            interpretation=result.get(
                "interpretation", ""
            ),
        )

    except Exception as exc:

        logger.exception(
            "[ERROR] API response construction failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Could not construct analysis response: {exc}"
        )

    # -----------------------------------------------------
    # STEP 7: SAVE ANALYSIS TO SUPABASE
    # -----------------------------------------------------

    try:

        from backend.database.supabase_db import (
            save_analysis,
        )

        save_start = time.perf_counter()

        await save_analysis(
            user_id,
            filename,
            result
        )

        logger.info(
            "[TIMING] Supabase save: %.2f seconds",
            time.perf_counter() - save_start
        )

    except Exception:

        logger.exception(
            "[WARNING] History save failed (non-blocking)"
        )

    # -----------------------------------------------------
    # STEP 8: TOTAL REQUEST TIME
    # -----------------------------------------------------

    logger.info(
        "[TIMING] Total request: %.2f seconds",
        time.perf_counter() - total_start
    )

    logger.info(
        "[ANALYSIS] Completed successfully for %s",
        filename
    )

    return response


# =========================================================
# HEALTH CHECK
# =========================================================

@router.get("/health")
async def health_check(request: Request):

    return {
        "status": "healthy",

        "nlp_loaded": (
            request.app.state.nlp is not None
        ),

        "embedder_loaded": (
            request.app.state.embedder is not None
        ),
    }


# =========================================================
# USER HISTORY
# =========================================================

@router.get("/history")
async def get_history(
    user_id: str = Depends(get_current_user)
):

    from backend.database.supabase_db import (
        get_user_history,
    )

    try:

        return await get_user_history(user_id)

    except Exception as exc:

        logger.exception(
            "[ERROR] History fetch failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Could not load history: {exc}"
        )


# =========================================================
# DELETE HISTORY ENTRY
# =========================================================

@router.delete("/history/{analysis_id}")
async def delete_history_entry(
    analysis_id: str,
    user_id: str = Depends(get_current_user),
):

    from backend.database.supabase_db import (
        delete_analysis,
    )

    try:

        success = await delete_analysis(
            analysis_id,
            user_id
        )

        if not success:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Analysis not found or not owned by this user."
                )
            )

        return {
            "status": "deleted",
            "id": analysis_id
        }

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "[ERROR] History deletion failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Could not delete: {exc}"
        )


# =========================================================
# GENERATE PDF REPORT
# =========================================================

@router.post("/generate-pdf")
async def generate_pdf(
    data: AnalysisResponse,
    user_id: str = Depends(get_current_user),
):

    from backend.services.report_generator import (
        generate_html_reports,
    )

    from backend.services.pdf_export import (
        generate_combined_pdf,
    )

    from fastapi.responses import Response

    try:

        html_docs = generate_html_reports(
            data.model_dump()
        )

        pdf_bytes = generate_combined_pdf(
            html_docs
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    "attachment; filename=ats_report.pdf"
                )
            }
        )

    except Exception as exc:

        logger.exception(
            "[ERROR] PDF generation failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate PDF: {exc}"
        )


# =========================================================
# GENERATE HISTORY PDF
# =========================================================

@router.get("/history/{analysis_id}/pdf")
async def generate_history_pdf(
    analysis_id: str,
    user_id: str = Depends(get_current_user),
):

    from backend.database.supabase_db import (
        get_user_history,
    )

    from backend.services.report_generator import (
        generate_html_reports,
    )

    from backend.services.pdf_export import (
        generate_combined_pdf,
    )

    from fastapi.responses import Response

    try:

        history = await get_user_history(user_id)

        analysis_data = next(
            (
                item["analysis_result"]
                for item in history
                if item["id"] == analysis_id
            ),
            None
        )

        if not analysis_data:

            raise HTTPException(
                status_code=404,
                detail="Analysis not found"
            )

        html_docs = generate_html_reports(
            analysis_data
        )

        pdf_bytes = generate_combined_pdf(
            html_docs
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    f"attachment; filename=ats_report_{analysis_id}.pdf"
                )
            }
        )

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "[ERROR] History PDF generation failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate PDF for history: {exc}"
        )
