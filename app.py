# -*- coding: utf-8 -*-
import os
import json
import requests
import urllib.parse
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
# from dotenv import load_dotenv # No es necesario para Vercel

# --- Configuración Inicial ---
# load_dotenv() # Puedes descomentar para desarrollo local si usas .env
app = Flask(__name__) # Instancia de Flask disponible globalmente
CORS(app) # Habilitar CORS para todas las rutas

# --- Configuración de Gemini ---
gemini_configured = False
model = None
# *** USANDO EL MODELO SOLICITADO ***
TARGET_GEMINI_MODEL = 'gemini-2.5-pro-exp-03-25'

try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        print(f"INFO: Initializing Gemini model: {TARGET_GEMINI_MODEL}...")
        model = genai.GenerativeModel(TARGET_GEMINI_MODEL)
        gemini_configured = True
        print("INFO: Gemini model initialized.")
    else:
        print("WARNING: Environment variable 'GOOGLE_API_KEY' NOT found.")
        print("WARNING: Gemini analysis will not work.")
except Exception as e:
    print(f"ERROR: An error occurred configuring Gemini: {e}")
    gemini_configured = False

# --- Funciones de Lógica (Helper Functions) ---
# (Son las mismas funciones que antes: call_ensembl_vep, run_initial_gemini_analysis, run_final_gemini_interpretation)
# (Se mantienen los prints dentro de estas funciones para depuración si es necesario)
def call_ensembl_vep(query_identifier):
    """Calls the Ensembl VEP API and returns data or an error message."""
    print(f"DEBUG: --- 1. Querying Ensembl VEP for: {query_identifier} ---")
    server_vep = "https://rest.ensembl.org"
    encoded_query = urllib.parse.quote(query_identifier)
    ext_vep_base = f"/vep/human/hgvs/{encoded_query}"
    dbnsfp_fields_extended = [
        "gnomAD_exomes_AF", "gnomAD_genomes_AF", "SIFT_pred", "Polyphen2_HDIV_pred",
        "MutationTaster_pred", "FATHMM_pred", "MetaSVM_pred", "M-CAP_pred",
        "PrimateAI_pred", "BayesDel_addAF_pred", "GERP++_RS", "clinvar_clnsig",
        "RVIS_EVS", "gnomAD_exomes_pLI", "gnomad_genomes_pLI"
    ]
    dbnsfp_param = f"dbNSFP={','.join(dbnsfp_fields_extended)}"
    optional_params_extended = [
        "hgvs=1", "protein=1", "uniprot=1", "canonical=1", "domains=1", "numbers=1",
        "variant_class=1", "Conservation=1", "LoF=1", "dbscSNV=1", "SpliceAI=1",
        "REVEL=1", "CADD=1", "ClinPred=1", "Mastermind=1", "DisGeNET=1",
        "Phenotypes=1", "LOEUF=1", "EVE=1", "AlphaMissense=1",
        dbnsfp_param, "mane=1", "clinvar_summary=1"
    ]
    params_string = "&".join(optional_params_extended)
    full_url = f"{server_vep}{ext_vep_base}?{params_string}"
    vep_timeout = 300

    try:
        print(f"DEBUG: Making GET request to Ensembl VEP (timeout={vep_timeout}s)... URL: {full_url}")
        response = requests.get(full_url, headers={"Content-Type": "application/json", "Accept": "application/json"}, timeout=vep_timeout)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            if len(data) > 1: print(f"WARN: Ensembl VEP returned {len(data)} results. Using the first one.")
            print("DEBUG: ✅ Ensembl VEP response received successfully.")
            return data[0], None
        elif data and isinstance(data, list) and len(data) == 0:
            error_msg = f"Ensembl VEP found no results (empty list) for '{query_identifier}'."
            print(f"ERROR: {error_msg}")
            return None, error_msg
        else:
            error_msg = f"Unexpected response format from Ensembl VEP: {str(data)[:300]}..."
            print(f"ERROR: {error_msg}")
            return None, error_msg
    except requests.exceptions.Timeout:
        error_msg = f"The request to Ensembl VEP timed out after {vep_timeout} seconds."
        print(f"ERROR: {error_msg}")
        return None, error_msg
    except requests.exceptions.RequestException as e:
        error_msg = f"Error during Ensembl VEP request: {e}"
        print(f"ERROR: {error_msg}")
        error_detail = ""
        status_code = 502
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            try: error_detail = json.dumps(e.response.json(), indent=2)
            except json.JSONDecodeError: error_detail = e.response.text
            print(f"ERROR Detail (Status {status_code}): {error_detail[:500]}")
        return None, f"{error_msg} (Status: {status_code}, Detail: {error_detail[:100]}...)"
    except Exception as e_general:
        error_msg = f"Unexpected error processing Ensembl VEP response: {e_general}"
        print(f"ERROR: {error_msg}")
        return None, error_msg

def run_initial_gemini_analysis(query_identifier, decoded_vep):
    """Performs the first Gemini call for quantitative classification."""
    global model, gemini_configured, TARGET_GEMINI_MODEL
    if not gemini_configured or not model:
        print("ERROR: Cannot run Gemini analysis, model not configured.")
        return None, "Gemini model is not configured on the backend."

    print(f"DEBUG: \n--- 2. Performing Initial Quantitative ACMG Analysis with Gemini ({TARGET_GEMINI_MODEL}, T=0.1) --- ")
    try:
        # El prompt detallado se mantiene igual que antes
        prompt_vep_quant_acmg_freq_assume = f"""
## ROLE AND GOAL
You are 'Variant Analyst AI', an expert system specialized in classifying human genetic sequence variants (SNVs and small indels) associated with Mendelian disorders. Your primary goal is to provide accurate, evidence-based classifications strictly following the quantitative point-based interpretation framework derived from the ACMG/AMP guidelines and subsequent ClinGen refinements.[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] You will receive variant annotation data from `VEP` and possibly minimal clinical context (if available). You must adhere to established best practices for variant interpretation, ensuring consistency, rigor, and transparency. **Your analysis MUST be based *solely* on the provided VEP JSON data, with one specific exception regarding missing frequency data as detailed below.**

## KNOWLEDGE BASE & METHODOLOGY
1.  **Foundation:** Use the 2015 ACMG/AMP "Standards and guidelines for the interpretation of sequence variants" (Richards et al., Genet Med 2015) criteria definitions as the starting point.[15, 16, 17, 18]
2.  **ClinGen Refinements:** Critically incorporate published recommendations and specifications from the ClinGen Sequence Variant Interpretation (SVI) working group and relevant Variant Curation Expert Panels (VCEPs) where applicable as general principles.[19, 1, 20, 21, 22, 23, 24, 25, 6, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37] This includes, but is not limited to:
    * **PVS1** criteria specifications (e.g., Abou Tayoun et al. recommendations, considerations for NMD, use of decision trees).[21, 24, 27, 38, 28, 39, 40, 41, 42]
    * **PS2/PM6** specifications for de novo variants (requiring parental identity confirmation for full-strength PS2).[43, 21, 38, 40]
    * **PM3** specifications for recessive disorders (in trans with a known pathogenic variant).[21, 39, 40]
    * **PP3/BP4** calibration for computational evidence (e.g., Pejaver et al., using calibrated tools like REVEL, BayesDel, AlphaMissense with defined score thresholds for Supporting, Moderate, Strong, or Very Strong strengths). Use a single well-calibrated tool's score **if available in the VEP JSON.**[2, 22, 26, 28, 44, 39, 40, 45, 35, 46, 12, 47, 48]
    * **BA1/BS1/PM2** allele frequency thresholds based on gnomAD data, appropriate population context, and calculations considering disease prevalence, penetrance, and heterogeneity.[43, 20, 21, 49, 38, 44, 39, 50, 40, 51] Note PM2 may be downgraded (e.g., PM2_Supporting) in some contexts.[39, 40] **MODIFIED RULE: If frequency data (e.g., gnomAD `af` in `colocated_variants` or `dbNSFP`) is absent or null in the VEP JSON, assume the variant is absent or extremely rare in the represented populations for the purpose of evaluating PM2 (absence/rarity supports pathogenicity). Do NOT apply BA1 or BS1 based on missing frequency data.** State this assumption clearly in your justification.
    * **PS3/BS3** for functional evidence, requiring validated assays **present in the VEP JSON** and distinguishing from direct RNA splicing assays.[21, 24, 28, 40, 45, 52]
    * **Splicing criteria** recommendations (e.g., application of PVS1_Strength, BP7 based on RNA data; PS1 based on predicted similarity; use of tools like SpliceAI with calibrated cutoffs **if SpliceAI data is present in VEP JSON**).[2, 24, 28, 39, 40, 45]
    * Awareness of criteria deemed inapplicable or requiring caution in certain contexts (e.g., PS4 proband counting in low penetrance [53, 38, 39, 40]; PP1/BS4 segregation in low penetrance [38, 40, 45, 52]; PM1 in genes with benign variants in functional domains [54, 38, 40]; PP5/BP6 reliance on external databases often discouraged [21, 55, 38, 8, 40]).
3.  **Quantitative Point-Based System (Primary Method):** Employ a quantitative point-based system consistent with the Bayesian framework described by Tavtigian et al. (Genet Med 2018; Hum Mutat 2020) as the **sole method** for combining evidence and determining the final classification.[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] This system replaces the qualitative combination rules from the 2015 guidelines.
    * **Point Allocation (Standard):** Pathogenic: PVS=+8, PS=+4, PM=+2, PP=+1. Benign: BA1/BS=-4, BP=-1. (Note: Strengths and corresponding points can be modified based on ClinGen specifications or specific evidence calibration [49, 26, 27, 39, 40, 45]).
    * **Classification Thresholds:**
        * Pathogenic ≥ +10 points
        * Likely Pathogenic +6 to +9 points
        * VUS 0 to +5 points
        * Likely Benign -1 to -6 points
        * Benign ≤ -7 points
4.  **Evidence Strength Modification:** Modify a criterion’s strength (and thus points) only if explicitly supported by ClinGen guidance, VCEP specifications, or robust, calibrated data **present in the VEP JSON.**[49, 26, 27, 8, 39, 40, 45] Provide explicit justification for any modification.
5.  **Handling Conflicting Evidence:** The point system inherently handles conflicting evidence by summing positive (pathogenic) and negative (benign) points for a net score.[2, 3, 4, 5, 6, 8, 10, 11, 12]
6.  **Scope Focus:** Prioritize variants related to Mendelian diseases, typically assuming high penetrance unless otherwise specified.[15, 17, 21] Acknowledge limitations when interpreting variants potentially associated with complex traits, low/moderate penetrance, or risk alleles, noting these may require different frameworks (e.g., ClinGen Low Penetrance/Risk Allele WG efforts).[43, 56, 20, 57, 58, 59, 60, 61, 62, 63, 64]

## INPUT DATA SPECIFICATION
You will receive variant information and structured annotation data retrieved via the `VEP` API for the query: '{query_identifier}'. Essential data fields expected are within the following JSON object:
```json
{json.dumps(decoded_vep, indent=2, default=str)}
```
Key fields to look for include:
* **Variant details:** `input`, `seq_region_name`, `start`, `end`, `allele_string`, `variant_class`.
* **Gene context:** Within `transcript_consequences`: `gene_symbol`, `gene_id`, potentially `biotype`. Look for established mechanism (e.g., LoF) based on gene knowledge.
* **Population Frequencies:** Within `colocated_variants`: Check entries with `source` = 'gnomADe'/'gnomADg', look for `af` (allele frequency). Also check `dbNSFP` fields like `gnomAD_exomes_AF`, `gnomAD_genomes_AF` **if requested and present in the JSON.**
* **Clinical Significance:** Within `colocated_variants`: Check entries with `source` = 'ClinVar', look for `clin_sig`, `review_status`. Also check `clinvar_summary` **if requested and present.**
* **Computational Predictions:**
    * Within `transcript_consequences`: `sift_prediction`, `sift_score`, `polyphen_prediction`, `polyphen_score`.
    * Within `plugin_output` (if requested): `REVEL`, `CADD`, `SpliceAI`, `AlphaMissense`, `ClinPred`.
    * Within `dbNSFP` (if requested): `REVEL_score`, `BayesDel_addAF_pred`, `CADD_phred`, `ClinPred_score`, etc. **Focus on calibrated scores ONLY IF PRESENT in the JSON.**
    * Conservation scores: `dbNSFP.gerp++_rs` or within `Conservation` plugin output **if requested and present.**
* **Variant Effect:** Within `transcript_consequences`: `consequence_terms`, `impact`, `hgvsc`, `hgvsp`, `lof`, `lof_flags`, `lof_filter`, `domain`.
* **Minimal Clinical Context (Optional):** None provided by default in this setup.

## TASK EXECUTION PROCESS
1.  **Data Review:** Systematically analyze all provided annotation data for the variant **strictly from the VEP JSON above.** Note any requested fields (e.g., specific predictors, gnomAD frequencies) that are missing in the JSON output.
2.  **Evidence Evaluation:** For each potentially applicable ACMG/AMP criterion code (PVS1, PS1–PS4, PM1–PM6, PP1–PP5, BA1, BS1–BS4, BP1–BP7):
    * Determine applicability based **only** on variant type, gene context, and available data **found within the provided VEP JSON**.
    * Evaluate the strength based on the provided data, adhering strictly to the quantitative framework and ClinGen/VCEP specifications. **For frequency criteria (BA1, BS1, PM2): If relevant frequency data (e.g., gnomAD `af` in `colocated_variants` or `dbNSFP`) is explicitly present in the VEP JSON, use it** according to ClinGen thresholds. **If such data is completely absent or null in the VEP JSON, assume the variant is absent/rare in the population** and apply PM2_Supporting (or stronger PM2 if other pathogenic criteria are met and the gene mechanism is LoF), but **do NOT apply BA1 or BS1 based on this assumption.** Clearly state that this assumption was made due to missing data. For all other criteria (computational predictors PP3/BP4, functional data PS3/BS3, etc.), only apply them if the relevant scores/data are explicitly present in the VEP JSON.
    * Assign the corresponding points (positive for pathogenic, negative for benign) based on the determined strength for the criteria that could be assessed with the available data.[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
3.  **Evidence Combination & Net Score:** Sum the points for all applied pathogenic criteria and benign criteria to produce a single net score.
4.  **Classification:** Determine the final classification (Pathogenic, Likely Pathogenic, VUS, Likely Benign, Benign) based **only** on the net points and the established thresholds.[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
5.  **VUS Sub-tiering:** If the classification is VUS (0-5 points), sub-classify based on points: VUS-Low (0-1 points), VUS-Mid (2-3 points), VUS-High (4-5 points).[1, 5, 6, 65, 9, 10, 11, 66, 47]

## OUTPUT REQUIREMENTS
Provide a structured response containing:
1.  **Final Classification:** Clearly state the final classification (e.g., "Likely Pathogenic"). If VUS, include the sub-tier and total points (e.g., "Variant of Uncertain Significance (VUS-Mid, 3 points)").
2.  **Criteria Summary & Justification:**
    * List **each** ACMG/AMP criterion code that was **applied**.
    * For each applied criterion, state its assigned **strength** (e.g., Moderate, Strong) and **points** (e.g., PM=+2, BS=-4).
    * Provide a **concise explanation** linking the criterion to the **specific data** from the `VEP` input JSON that justifies its application and strength (e.g., "PM2 (+1 punto): Datos de frecuencia gnomAD ausentes en VEP JSON; se asume ausencia/rareza en la población para aplicar PM2_Supporting (según regla especificada)"). Use specific values from the JSON if available for other criteria.
    * Explicitly mention any **strength modifications** applied based on ClinGen guidance or data calibration (e.g., "PP3 (+1 punto): Puntuación REVEL = 0.92 (obtenida de `plugin_output` o `dbNSFP`) cumple el umbral calibrado para fuerza Supporting según Pejaver et al. 2022").
    * List any significant criteria that were **considered but NOT applied**, explaining **specifically why the data in the VEP JSON was insufficient or absent** (e.g., "PVS1 no aplicado: Aunque `consequence_terms` incluye 'frameshift_variant', ocurre en el último exón (`transcript_consequences.exon` = X/X) y `lof_flags` indica posible escape a NMD según recomendaciones de Abou Tayoun et al.", "PS3 no aplicado: No se proporcionaron datos de estudios funcionales validados en el input JSON", "BA1/BS1 no aplicado: Datos de frecuencia gnomAD (`af`) no encontrados en `colocated_variants` ni en `dbNSFP` dentro del JSON de VEP proporcionado O (si se aplicó PM2 por ausencia) no aplicables por falta de datos de alta frecuencia.").
3.  **Net Score Calculation:** Show the sum of points from pathogenic vs. benign criteria (e.g., “Suma de puntos de criterios patogénicos = +7, Suma de puntos de criterios benignos = -1, Puntuación Neta = +6”).
4.  **Recommendations / Data Gaps (If Applicable):** If classification remains uncertain (VUS) or if key data **required for specific criteria** were missing from the VEP JSON output (like predictors, ClinVar info, etc.), state this clearly. Note if frequency data was missing and PM2 was applied based on assumption. Suggest non-clinical follow-up actions (e.g., “La ausencia de predictores clave (REVEL, AlphaMissense) en la salida de VEP limita la evaluación PP3/BP4. Un ensayo funcional validado podría aportar evidencia PS3/BS3,” “Estudio de los padres podría evaluar estado de novo (PS2/PM6)”).
5.  **Statement of Limitations:**
    * Highlight that this classification is an *in silico* assessment based **solely** on the public data retrieved from Ensembl VEP **as present in the input JSON** and established guidelines.
    * State that uncertain or missing information within the VEP JSON output (e.g., lack of family segregation data, unpublished functional data, missing annotations in source databases queried by VEP) may significantly affect the classification. Note specifically if the PM2 application relied on an assumption due to missing frequency data.
    * Emphasize that the interpretation requires review and final sign-off by a qualified human expert considering the full clinical context.

## CONSTRAINTS AND INTERACTION STYLE
* **Data Adherence (with Frequency Exception):** Base your classification *exclusively* on the data present within the provided VEP JSON, **with the specific exception for population frequency:** if frequency data (e.g., gnomAD `af`) is missing from the VEP JSON, **assume absence/rarity for applying PM2 (state this assumption).** Do not apply BA1/BS1 based on missing data. For all other criteria, do NOT invent data, infer information not explicitly present (e.g., functional impact without functional data), or make assumptions beyond the provided annotations. If data for other criteria is missing from the JSON, the criterion cannot be applied based on that specific piece of evidence.
* **No Clinical Advice:** Do not provide clinical recommendations or medical advice.
* **Transparency:** Be explicit about the rules, thresholds, and points used. Maintain an objective, analytical, and professional tone.
* **Conflict Handling:** If the input data JSON contains contradictory fields (e.g., conflicting ClinVar entries), note this in the justification but proceed with classification based on the available evidence and point system.

**IMPORTANTE:** Por favor, genera la respuesta final completa exclusivamente en **español**.
"""
        gemini_timeout = 480
        generation_config = genai.types.GenerationConfig(temperature=0.1)

        print(f"DEBUG: Sending initial analysis request to Gemini (timeout={gemini_timeout}s)...")
        response_gemini = model.generate_content(
            prompt_vep_quant_acmg_freq_assume,
            generation_config=generation_config,
            request_options={'timeout': gemini_timeout}
        )
        print("DEBUG: --- Results from Initial Gemini Analysis ---")

        try:
            analysis_output_markdown = response_gemini.text
            print("DEBUG: ✅ Initial analysis from Gemini received.")
            return analysis_output_markdown, None
        except ValueError:
            error_msg = f"Gemini response (initial) was blocked. Feedback: {response_gemini.prompt_feedback}"
            print(f"ERROR: {error_msg}")
            return None, error_msg
        except Exception as e_resp:
            error_msg = f"Unexpected error processing initial Gemini response: {e_resp}"
            print(f"ERROR: {error_msg}")
            return None, error_msg

    except Exception as e_gemini_call:
        error_msg = f"Error calling Gemini for initial analysis: {e_gemini_call}"
        print(f"ERROR: {error_msg}")
        return None, error_msg

def run_final_gemini_interpretation(query_identifier, gene_name, initial_analysis_markdown, clinical_info):
    """Performs the second Gemini call integrating clinical data."""
    global model, gemini_configured, TARGET_GEMINI_MODEL
    if not gemini_configured or not model:
        print("ERROR: Cannot run Gemini interpretation, model not configured.")
        return None, "Gemini model is not configured on the backend."
    if not initial_analysis_markdown:
        print("ERROR: Missing initial analysis result for final interpretation.")
        return None, "Missing initial analysis result for final interpretation."

    print(f"DEBUG: \n--- 3. Performing Final Interpretation with Clinical Data ({TARGET_GEMINI_MODEL}, T=0.1) --- ")
    try:
        # El prompt detallado se mantiene igual que antes
        prompt_final_interpretation = f"""
## ROLE AND GOAL
You are 'Variant Interpretation AI', an expert system assisting clinical geneticists. Your goal is to integrate a **pre-computed variant classification** (based on ACMG quantitative points derived from VEP data) with **patient-specific clinical information** to provide a final interpretation regarding the variant's potential causal role in the patient's phenotype.

## INPUTS
1.  **Variant Information:**
    * Query: '{query_identifier}'
    * Gene: '{gene_name}'
2.  **Initial Variant Classification Report (ACMG Quantitative Points from VEP):**
    ```markdown
    {initial_analysis_markdown}
    ```
3.  **Patient Clinical Information:**
    ```text
    {clinical_info}
    ```

## TASK
Based **only** on the information provided above (initial classification report and clinical information):
1.  **Summarize:** Briefly summarize the key clinical features provided for the patient.
2.  **Summarize:** Briefly summarize the initial ACMG classification result and the main evidence points (pathogenic and benign) for the variant '{query_identifier}'.
3.  **Correlation:** Evaluate the correlation between the patient's phenotype and the known function/disease association of the gene '{gene_name}'. Does the phenotype fit? Are there specific features that strongly match or mismatch?
4.  **Integration & Interpretation:** Combine the variant classification strength (Pathogenic, Likely Pathogenic, VUS, Likely Benign, Benign based on the point score in the initial report) with the clinical correlation.
    * If the variant is Pathogenic/Likely Pathogenic AND the phenotype strongly correlates with the gene, interpret the variant as likely causal or contributing to the patient's presentation.
    * If the variant is Pathogenic/Likely Pathogenic BUT the phenotype DOES NOT correlate well, discuss this discrepancy. Could it be a secondary finding, reduced penetrance, or an atypical presentation?
    * If the variant is VUS (mention sub-tier if available), discuss the uncertainty. Does the clinical picture increase or decrease the suspicion for this variant/gene? Are there features that might support upgrading/downgrading the VUS if further evidence becomes available? Emphasize that a VUS cannot be used alone for clinical decisions.
    * If the variant is Likely Benign/Benign, interpret it as unlikely to be causally related to the core Mendelian phenotype described, even if there is some clinical overlap (unless the phenotype is extremely non-specific).
5.  **Final Conclusion:** Provide a concise conclusion about the likely role of the variant in this specific patient's case, highlighting any remaining uncertainties or necessary considerations.

## CONSTRAINTS
* Base the interpretation **strictly** on the provided initial report and clinical information. Do not re-evaluate ACMG criteria or consult external databases again.
* Focus on integrating the two pieces of information.
* Maintain a cautious and objective tone, especially regarding VUS.
* Do not provide medical advice or treatment recommendations.

**IMPORTANTE:** Por favor, genera la respuesta final completa exclusivamente en **español**.
"""
        gemini_timeout = 300
        generation_config = genai.types.GenerationConfig(temperature=0.1)

        print(f"DEBUG: Sending final interpretation request to Gemini (timeout={gemini_timeout}s)...")
        response_gemini = model.generate_content(
            prompt_final_interpretation,
            generation_config=generation_config,
            request_options={'timeout': gemini_timeout}
        )
        print("DEBUG: --- Results from Final Gemini Interpretation ---")

        try:
            final_interpretation_markdown = response_gemini.text
            print("DEBUG: ✅ Final interpretation from Gemini received.")
            return final_interpretation_markdown, None
        except ValueError:
            error_msg = f"Gemini response (final interpretation) was blocked. Feedback: {response_gemini.prompt_feedback}"
            print(f"ERROR: {error_msg}")
            return None, error_msg
        except Exception as e_resp:
            error_msg = f"Unexpected error processing final Gemini response: {e_resp}"
            print(f"ERROR: {error_msg}")
            return None, error_msg

    except Exception as e_gemini_call:
        error_msg = f"Error calling Gemini for final interpretation: {e_gemini_call}"
        print(f"ERROR: {error_msg}")
        return None, error_msg


# --- Rutas de la API Flask ---

# Define la ruta para el análisis inicial
@app.route('/api/analyze', methods=['POST'])
def analyze_variant():
    """Endpoint para el análisis inicial (VEP + Gemini inicial)."""
    print("DEBUG: Entering /api/analyze route") # Print para confirmar entrada a la ruta
    if not request.is_json:
        print("ERROR: Request is not JSON")
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    data = request.get_json()
    print(f"DEBUG: Received data for /api/analyze: {data}") # Print datos recibidos

    gene = data.get('gene')
    transcript = data.get('transcript')
    cdna = data.get('cdna')

    # Validación
    errors = []
    if not gene: errors.append("El campo 'Gen' es obligatorio.")
    if not transcript: errors.append("El campo 'Transcrito Ref.' es obligatorio.")
    if not cdna: errors.append("El campo 'Variante cDNA' es obligatorio.")
    elif not cdna.startswith('c.'): errors.append("El formato de 'Variante cDNA' debe empezar con 'c.'")

    if errors:
        error_message = f"Errores de validación: {'; '.join(errors)}"
        print(f"WARNING: Validation failed for /api/analyze: {error_message}")
        return jsonify({"status": "error", "message": error_message}), 400

    # Si la validación pasa
    print("DEBUG: Validation passed for /api/analyze")
    query_identifier = f"{transcript}:{cdna}"

    # 1. Llamar a VEP
    vep_data, vep_error = call_ensembl_vep(query_identifier)
    if vep_error:
        print(f"ERROR: VEP call failed: {vep_error}")
        return jsonify({"status": "error", "message": f"Error al contactar Ensembl VEP: {vep_error}"}), 502
    if not vep_data:
        print("ERROR: No valid data received from VEP.")
        return jsonify({"status": "error", "message": "No se obtuvieron datos válidos de Ensembl VEP."}), 404

    # 2. Llamar a Gemini (Análisis Inicial)
    initial_markdown, gemini_error = run_initial_gemini_analysis(query_identifier, vep_data)
    if gemini_error:
        print(f"ERROR: Gemini initial analysis failed: {gemini_error}")
        return jsonify({"status": "error", "message": f"Error durante el análisis inicial con IA: {gemini_error}"}), 500
    if not initial_markdown:
        print("ERROR: Gemini initial analysis did not produce results.")
        return jsonify({"status": "error", "message": "El análisis inicial con IA no produjo resultados."}), 500

    # 3. Devolver resultado inicial exitoso
    print("DEBUG: --- /api/analyze processing completed successfully ---")
    return jsonify({"status": "success", "markdown_result": initial_markdown})

# Define la ruta para la interpretación final
@app.route('/api/interpret_clinical', methods=['POST'])
def interpret_clinical_data():
    """Endpoint para la interpretación final con datos clínicos."""
    print("DEBUG: Entering /api/interpret_clinical route") # Print para confirmar entrada
    if not request.is_json:
        print("ERROR: Request is not JSON for /api/interpret_clinical")
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    data = request.get_json()
    print(f"DEBUG: Received data for /api/interpret_clinical: {data}") # Print datos recibidos

    gene = data.get('gene')
    transcript = data.get('transcript') # Recibir de nuevo para reconstruir query_id
    cdna = data.get('cdna')             # Recibir de nuevo para reconstruir query_id
    initial_markdown = data.get('initial_markdown')
    clinical_info = data.get('clinical_info')

    # Reconstruir query_identifier
    query_identifier = f"{transcript}:{cdna}" if transcript and cdna else None

    # Validación
    errors = []
    if not gene: errors.append("Falta el campo 'Gen'.")
    if not query_identifier: errors.append("Faltan 'Transcrito Ref.' o 'Variante cDNA' para reconstruir el identificador.")
    if not initial_markdown: errors.append("Falta el resultado del análisis inicial.")
    if not clinical_info: errors.append("Falta la información clínica.")

    if errors:
        error_message = f"Errores de validación: {'; '.join(errors)}"
        print(f"WARNING: Validation failed for /api/interpret_clinical: {error_message}")
        return jsonify({"status": "error", "message": error_message}), 400

    # Si la validación pasa
    print("DEBUG: Validation passed for /api/interpret_clinical")
    # Llamar a Gemini (Interpretación final)
    final_interpretation, gemini_error = run_final_gemini_interpretation(query_identifier, gene, initial_markdown, clinical_info)

    if gemini_error:
        print(f"ERROR: Gemini final interpretation failed: {gemini_error}")
        return jsonify({"status": "error", "message": f"Error durante la interpretación final con IA: {gemini_error}"}), 500
    if not final_interpretation:
        print("ERROR: Gemini final interpretation did not produce results.")
        return jsonify({"status": "error", "message": "La interpretación final con IA no produjo resultados."}), 500

    # Devolver resultado final exitoso
    print("DEBUG: --- /api/interpret_clinical processing completed successfully ---")
    return jsonify({"status": "success", "final_interpretation": final_interpretation})

# No se necesita app.run() para Vercel
