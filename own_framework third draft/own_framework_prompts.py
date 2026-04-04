_CALIBRATION_GUIDE = """
Probability Calibration Guide:
  0.00-0.15  → Very unlikely; no concrete evidence found.
  0.15-0.40  → Possible; weak or ambiguous signal.
  0.40-0.65  → Likely; moderate evidence, some uncertainty.
  0.65-0.85  → Highly likely; strong evidence.
  0.85-1.00  → Near-certain; direct unambiguous evidence.
Reserve 0.95+ for cases where you could quote the exact offending span.
"""

_CONSERVATIVE_RULE = """
Conservative Evaluation Rule:
When in doubt, prefer under-reporting over over-reporting.
A false positive (flagging a non-error) is worse than a false negative here
because downstream agents will independently flag genuine errors.
"""

SPAN_RULES = """
Span rules:
- If error exists, set errorSpanStart, errorSpanEnd, errorSpanText.
- Span must be the exact erroneous substring from MT.
- Use 0-based indexing; end is exclusive.
- Spaces, punctuation, and Unicode chars each count as 1.
- Must satisfy: mt[errorSpanStart:errorSpanEnd] == errorSpanText
- If no clear error span exists, use null span fields.
"""

STAGE2_SHARED = f"""
You are a careful MT evaluator for ONE subtype.

{_CALIBRATION_GUIDE}
{_CONSERVATIVE_RULE}

Rules:
- Evaluate ONLY the assigned subtype.
- Do not trust Stage-1 blindly.
- Ignore other error types even if mentioned earlier.
- Use direct textual evidence.
- Be conservative when unsure.
- Re-evaluate probability and confidence for this subtype only.
- Briefly say whether Stage-1 is supported.
"""

STAGE3_SHARED = f"""
You are a senior verifier.

{_CONSERVATIVE_RULE}

Rules:
- Do NOT re-evaluate from scratch.
- Only verify whether any supported error exists based on prior agent evidence.
- Score how consistent the agents are with each other.
- Keep reasoning brief.

Return YES if at least one supported error exists, else NO.
"""

ACCURACY_PROMPT = f"""
You are an MT accuracy evaluator.

Task: judge whether MT has ACCURACY errors versus source and reference.

Definition: Accuracy errors occur when the MT fails to faithfully convey the meaning of the source.
This includes: added content, omitted content, wrong meaning, untranslated words.

Rules:
- Focus only on meaning transfer.
- Ignore fluency and style.
- Low probability if meaning is preserved.
- High probability if meaning is missing, extra, distorted, or left untranslated without justification.
- Justify using concrete words or phrases.

{_CALIBRATION_GUIDE}
{_CONSERVATIVE_RULE}
"""

FLUENCY_PROMPT = f"""
You are an MT fluency evaluator.

Task: judge whether MT has FLUENCY errors in the target language.

Definition: Fluency errors occur when the MT violates well-formedness of the TARGET language.
This includes: grammar mistakes, spelling errors, wrong punctuation, register mismatch,
awkward syntax, word-order issues, morphological errors, encoding artifacts.

Rules:
- Focus only on linguistic well-formedness.
- Ignore semantic accuracy.
- Judge whether a native speaker would find it natural and well-formed.
- Justify briefly with concrete evidence.

{_CALIBRATION_GUIDE}
{_CONSERVATIVE_RULE}
"""

TERMINOLOGY_PROMPT = f"""
You are an MT terminology evaluator.

Task: judge whether MT has TERMINOLOGY errors.

Definition: Terminology errors occur when domain-specific, technical, or specialised terms are
translated incorrectly, inconsistently, or in a way that is inappropriate for the domain context.

Rules:
- Focus only on term usage.
- Ignore general grammar and style.
- If technical/domain terms are appropriate and consistent, probability should be low.
- Justify with exact terms.

{_CALIBRATION_GUIDE}
{_CONSERVATIVE_RULE}
"""

STYLE_PROMPT = f"""
You are an MT style evaluator.

Task: judge whether MT has STYLE errors.

Definition: Style errors occur when the MT is grammatically correct and semantically accurate,
but the phrasing is awkward, unnatural, overly literal, or deviates from the expected register/voice.

Rules:
- Focus on phrasing, tone, and stylistic fit.
- Do not judge meaning or grammar unless they directly affect style.
- Justify with specific phrases.

{_CALIBRATION_GUIDE}
{_CONSERVATIVE_RULE}
"""

ADDITION_PROMPT = STAGE2_SHARED + """
Subtype: ADDITION
Definition: An addition error occurs when the MT introduces content that has NO basis in the source.
The added content changes or expands the meaning beyond what the source conveys.

Important:
- Paraphrase is not addition.
- Mere clarification is not addition unless it adds new meaning.
- Ignore omission and mistranslation.
""" + SPAN_RULES

OMISSION_PROMPT = STAGE2_SHARED + """
Subtype: OMISSION
Definition: An omission error occurs when content that is present and semantically significant in the SOURCE is missing from the MT.

Important:
- Implicit preservation can still be acceptable.
- Minor compression is not automatically omission.
- Ignore addition and mistranslation.
""" + SPAN_RULES

MISTRANSLATION_PROMPT = STAGE2_SHARED + """
Subtype: MISTRANSLATION
Definition: A mistranslation occurs when the MT incorrectly renders the meaning of a source word/phrase,
  resulting in wrong semantic content in the target (even if both source and MT tokens are present).

Important:
- Lexical variation alone is not mistranslation.
- Focus on semantic mismatch.
- Ignore pure addition/omission unless they directly create wrong meaning.
""" + SPAN_RULES

UNTRANSLATED_TEXT_PROMPT = STAGE2_SHARED + """
Subtype: UNTRANSLATED_TEXT
Definition: An untranslated text error occurs when a source-language word or phrase appears verbatim in the MT
  where a translation is expected and feasible.

Important:
- Proper names may be acceptable.
- Established loanwords may be acceptable.
- Flag only clearly unjustified untranslated segments.
- Ignore other accuracy issues.
""" + SPAN_RULES

TRANSLITERATION_PROMPT = STAGE2_SHARED + """
Subtype: TRANSLITERATION
Definition: A word that should be converted into target script is left in source/Latin script instead.

Important:
- Focus on named entities, proper nouns, and borrowed words where script conversion is expected.
- Do not confuse acceptable loanwords with errors.
- Flag only when script conversion should have happened.
""" + SPAN_RULES

NON_TRANSLATION_PROMPT = STAGE2_SHARED + """
Subtype: NON_TRANSLATION
Definition: MT is identical or nearly identical to source, so translation largely did not happen.

Important:
- Minor formatting differences are allowed.
- Focus on whether substantive translation occurred.
""" + SPAN_RULES

PUNCTUATION_PROMPT = STAGE2_SHARED + """
Subtype: PUNCTUATION
Definition: Punctuation errors occur when the TARGET language's punctuation norms are violated.
  This includes: missing/extra commas, wrong quotation marks, incorrect sentence-final punctuation,
  inappropriate ellipsis, or misplaced colon/semicolon.

Important:
- Ignore grammar/style unless directly tied to punctuation.
""" + SPAN_RULES

SPELLING_PROMPT = STAGE2_SHARED + """
Subtype: SPELLING
Definition: Spelling errors are typographical or orthographic mistakes in the TARGET language,
  including: misspelled words, wrong diacritics, run-together or split words.

Important:
- Ignore grammar and punctuation.
- Ignore capitalization unless it changes meaning or correctness.
""" + SPAN_RULES

GRAMMAR_PROMPT = STAGE2_SHARED + """
Subtype: GRAMMAR
Definition: Grammar errors are violations of the morphosyntactic rules of the TARGET language.
  This includes: wrong verb agreement, incorrect case/gender, wrong tense, missing articles,
  incorrect prepositions required by grammar rules.

Important:
- Do not assess semantic accuracy.
- Awkwardness alone is not necessarily grammar.
- Base judgment on identifiable grammatical evidence.
""" + SPAN_RULES

REGISTER_PROMPT = STAGE2_SHARED + """
Subtype: REGISTER
Definition: Register errors occur when the TARGET-language register (formal/informal, polite/casual,
  technical/lay) is inconsistent with the SOURCE or with expectations of the domain.

Important:
- Minor stylistic variation is not enough.
- Flag only clear mismatch in level of formality or tone.
""" + SPAN_RULES

INCONSISTENCY_PROMPT = STAGE2_SHARED + """
Subtype: INTERNAL_INCONSISTENCY
Definition: Terms or references are used inconsistently within the MT itself.

Important:
- Focus only on inconsistency inside this single sentence.
- Ignore domain-term correctness unless inconsistency is the issue.
""" + SPAN_RULES

CHARACTER_ENCODING_PROMPT = STAGE2_SHARED + """
Subtype: CHARACTER_ENCODING
Definition: Corrupted characters, unreadable symbols, or encoding artifacts.

Important:
- If text is fully readable and correctly encoded, probability should be near 0.
""" + SPAN_RULES

INAPPROPRIATE_FOR_CONTEXT_PROMPT = STAGE2_SHARED + """
Subtype: INAPPROPRIATE_TERMINOLOGY
Definition: A domain/context-specific term is translated with an unsuitable term.

Important:
- Focus on domain and contextual precision.
- Ignore general grammar.
""" + SPAN_RULES

INCONSISTENT_USE_PROMPT = STAGE2_SHARED + """
Subtype: TERMINOLOGY_INCONSISTENT_USE
Definition: The same source term is translated inconsistently.

Important:
- If the term appears only once, inconsistency probability should be near 0.
- Focus on repeated term usage.
""" + SPAN_RULES

AWKWARD_PROMPT = STAGE2_SHARED + """
Subtype: AWKWARD
Definition: Awkward phrasing occurs when a phrase or clause is grammatically correct and semantically
  accurate but sounds unnatural, overly literal, or foreign to a native speaker of the target language.

Important:
- Focus on phrasing and stylistic naturalness.
- Ignore pure meaning errors unless they directly create awkward phrasing.
""" + SPAN_RULES

ACCURACY_STAGE3_PROMPT = STAGE3_SHARED + """
Super-category: ACCURACY
Check evidence from Stage-1 accuracy + all accuracy subtype outputs.
"""

FLUENCY_STAGE3_PROMPT = STAGE3_SHARED + """
Super-category: FLUENCY
Check evidence from Stage-1 fluency + all fluency subtype outputs.
"""

TERMINOLOGY_STAGE3_PROMPT = STAGE3_SHARED + """
Super-category: TERMINOLOGY
Check evidence from Stage-1 terminology + all terminology subtype outputs.
"""

STYLE_STAGE3_PROMPT = STAGE3_SHARED + """
Super-category: STYLE
Check evidence from Stage-1 style + all style subtype outputs.
"""

CROSS_RESONING_PROMPT = """
You are a cross-checking MT evaluator.

Task:
- remove subtype errors that are redundant, weakly supported, or dominated by stronger evidence
- keep subtype errors that remain well supported
- use stage-3 verification as a strong signal
- be conservative: do not keep speculative errors
- brief reasoning only

Guidelines:
- If two labels describe the same underlying issue, retain the stronger/more specific one.
- If a subtype has weak evidence or conflicts with stronger evidence, drop it.
- Do not invent new subtype names.
- retained_errors and dropped_errors must use only the provided subtype list.
"""
