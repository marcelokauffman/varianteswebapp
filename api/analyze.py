# -*- coding: utf-8 -*-
# Importar bibliotecas necesarias
import os
import json
import requests
import urllib.parse
import re # Para extraer Post_P
from http.server import BaseHTTPRequestHandler
from io import BytesIO

# Importar y configurar Gemini
try:
    import google.generativeai as genai
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    TARGET_GEMINI_MODEL = 'gemini-2.5-pro-exp-03-25' # O el modelo estable que prefieras
    model = None
    gemini_configured = False
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(TARGET_GEMINI_MODEL)
        gemini_configured = True
        print("Gemini configured successfully.")
    else:
        print("ERROR: GOOGLE_API_KEY environment variable not found.")
except ImportError:
    genai = None; model = None; gemini_configured = False
    print("ERROR: google-generativeai library not found.")
except Exception as e:
    genai = None; model = None; gemini_configured = False
    print(f"ERROR configuring Gemini: {e}")

# --- Función para extraer Post_P ---
def extract_post_p(report_text):
    """Extrae el valor numérico de Post_P del texto del reporte."""
    if not report_text:
        return None
    try:
        # Buscar el valor numérico después de "Probabilidad Posterior (Post_P):"
        match = re.search(r"Probabilidad\s+Posterior\s+\(Post_P\):\s*([0-9.eE+-]+)", report_text, re.IGNORECASE)
        if match:
            post_p_value = float(match.group(1))
            # Validar rango
            if 0.0 <= post_p_value <= 1.0:
                return post_p_value
            else:
                print(f"WARN: Extracted Post_P value {post_p_value} out of range [0, 1].")
                return None
        else:
            print("WARN: Could not find Post_P value in Gemini report.")
            return None
    except Exception as e:
        print(f"Error extracting Post_P: {e}")
        return None

# --- Función Principal del Handler (Compatible con Vercel) ---
class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        """Handles POST requests to /api/analyze."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        response_data = {}
        status_code = 500 # Default a error interno

        try:
            input_data = json.loads(body.decode('utf-8'))
            gene_name_val = input_data.get('gene')
            transcript_id_val = input_data.get('transcript_id')
            cdna_variant_val = input_data.get('cdna_variant')

            # --- Validación ---
            errors = []
            if not gene_name_val: errors.append("El campo 'Gen' es obligatorio.")
            if not transcript_id_val: errors.append("El campo 'Transcrito Ref.' es obligatorio.")
            if not cdna_variant_val: errors.append("El campo 'Variante cDNA' es obligatorio.")
            elif not cdna_variant_val.startswith('c.'): errors.append("El formato de 'Variante cDNA' debe empezar con 'c.'")

            if errors:
                response_data = {'error': True, 'message': "Errores de validación: " + "; ".join(errors)}
                status_code = 400
            elif not gemini_configured or not model:
                 response_data = {'error': True, 'message': "Error interno: El modelo Gemini no está configurado."}
                 status_code = 500
            else:
                # --- Ejecutar el flujo de análisis ---
                results = self.run_analysis_pipeline(gene_name_val, transcript_id_val, cdna_variant_val)
                response_data = results
                status_code = 200 if not results.get('error') else results.get('status_code', 500)

        except json.JSONDecodeError:
            response_data = {'error': True, 'message': "Error: Cuerpo de la solicitud no es JSON válido."}
            status_code = 400
        except Exception as e:
            print(f"Unhandled Exception in POST handler: {e}")
            response_data = {'error': True, 'message': f"Error inesperado en el servidor: {e}"}
            status_code = 500

        # Enviar respuesta
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*') # Permitir CORS
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode('utf-8'))

    # --- Lógica de Análisis ---
    def run_analysis_pipeline(self, local_gene_name, local_transcript_id, local_cdna_variant):
        """Orquesta las llamadas a API y Gemini."""
        global model, TARGET_GEMINI_MODEL

        query_identifier = None
        decoded_vep = None
        vep_success = False
        acmg_bayesian_report = None
        posterior_probability = None
        pipeline_error_message = None
        pipeline_status_code = 500

        # --- Paso 1: Construir HGVS ---
        print(f"--- 1. Construyendo HGVS para Gen:{local_gene_name}, Tx:{local_transcript_id}, cDNA:{local_cdna_variant} ---")
        if local_transcript_id and local_cdna_variant:
             query_identifier = f"{local_transcript_id}:{local_cdna_variant}"
             print(f"Query VEP: {query_identifier}")
        else:
             pipeline_error_message = "Error interno: No se pudo construir identificador HGVS."
             return {'error': True, 'message': pipeline_error_message, 'status_code': 500}

        # --- Paso 2: VEP Call (Extendido) ---
        try:
            print(f"--- 2. Consultando VEP para: {query_identifier} ---")
            server_vep = "https://rest.ensembl.org"
            encoded_query_hgvs = urllib.parse.quote(query_identifier)
            # Parámetros VEP Extendidos
            dbnsfp_fields_extended = [
                "gnomAD_exomes_AF", "gnomAD_genomes_AF", "SIFT_pred", "Polyphen2_HDIV_pred",
                "MutationTaster_pred", "FATHMM_pred", "MetaSVM_pred", "M-CAP_pred",
                "PrimateAI_pred", "BayesDel_addAF_pred", "GERP++_RS", "clinvar_clnsig",
                "RVIS_EVS", "gnomAD_exomes_pLI", "gnomAD_genomes_pLI"
            ]
            dbnsfp_param = f"dbNSFP={','.join(dbnsfp_fields_extended)}"
            optional_params_extended = [
                "hgvs=1", "protein=1", "uniprot=1", "canonical=1", "domains=1", "numbers=1",
                "variant_class=1", "Conservation=1", "LoF=1", "dbscSNV=1", "SpliceAI=2",
                "REVEL=1", "CADD=1", "ClinPred=1", "Mastermind=1", "DisGeNET=1",
                "Phenotypes=1", "LOEUF=1", "EVE=1", "AlphaMissense=1",
                dbnsfp_param, "mane=1"
            ]
            params_string = "&".join(optional_params_extended)
            ext_vep = f"/vep/human/hgvs/{encoded_query_hgvs}?{params_string}"

            vep_timeout = 300 # 5 minutos
            print(f"Realizando llamada a VEP (timeout={vep_timeout}s)...")
            r_vep = requests.get(server_vep + ext_vep, headers={"Content-Type": "application/json", "Accept": "application/json"}, timeout=vep_timeout)
            r_vep.raise_for_status()
            decoded_vep = r_vep.json()
            print("VEP OK.")
            vep_success = True

        except requests.exceptions.Timeout:
            pipeline_error_message = f"Error: La consulta a VEP superó el tiempo límite ({vep_timeout}s)."
            pipeline_status_code = 504
            return {'error': True, 'message': pipeline_error_message, 'status_code': pipeline_status_code}
        except requests.exceptions.RequestException as e:
            print(f"Error en Paso 2 (VEP): {e}")
            error_detail = f"{e}"
            status_code_from_error = 503 # Default
            if hasattr(e, 'response') and e.response is not None:
                 status_code_from_error = e.response.status_code if e.response.status_code >= 400 else 503
                 try: error_detail += f" - Detalle: {json.dumps(e.response.json())}"
                 except json.JSONDecodeError: error_detail += f" - Contenido: {e.response.text}"
            pipeline_error_message = f"Error al consultar VEP: {error_detail}"
            pipeline_status_code = status_code_from_error
            return {'error': True, 'message': pipeline_error_message, 'status_code': pipeline_status_code}

        # --- Paso 3: Análisis Combinado con Gemini ---
        try:
            if vep_success and decoded_vep:
                print(f"--- 3. Generando Análisis Combinado con Gemini ({TARGET_GEMINI_MODEL}) ---")
                # Prompt Combinado y Refinado (usando el JSON completo de VEP)
                prompt_combined_analysis = f"""
                **Rol:** Eres un genetista clínico especializado... (como en la Celda 4 final)

                **Input:** La variante a analizar es '{query_identifier}' (gen '{local_gene_name}', basada en el transcrito de referencia proporcionado '{local_transcript_id}'). La evidencia disponible... incluye las siguientes anotaciones...:
                ```json
                {json.dumps(decoded_vep, indent=2)}
                ```

                **Tarea:** Realiza un análisis exhaustivo...

                **PARTE 1: Análisis y Clasificación ACMG Refinada**
                1.  **Identificación de Variante:** ...
                2.  **Aplicación de Criterios ACMG:** ... (Incluir lógica de frecuencia ausente/rara) ...
                3.  **Clasificación Cualitativa Final:** ...
                4.  **Nivel de Confianza:** ...
                5.  **Implicaciones Clínicas:** ... (Lenguaje médico) ...

                **PARTE 2: Análisis Bayesiano (Modelo Tavtigian/Quaio)**
                1.  **Criterios Usados:** ...
                2.  **Cálculo Bayesiano:** ... Asigna Odds... Calcula Combined_OP... Calcula Post_P... **Muestra el valor numérico de Post_P claramente.**
                3.  **Clasificación Bayesiana Final:** ...

                **Formato de Salida Requerido (en español y Markdown):** ... Incluye el descargo de responsabilidad al final.

                **Descargo de Responsabilidad Obligatorio:**
                **IMPORTANTE:** Esta clasificación e interpretación... (completo como antes) ... requiere una revisión experta completa.
                """
                gemini_timeout = 300 # 5 minutos
                print(f"Enviando solicitud a Gemini (timeout={gemini_timeout}s)...")
                response_combined = model.generate_content(prompt_combined_analysis, request_options={'timeout': gemini_timeout})

                if response_combined.parts:
                    acmg_bayesian_report = response_combined.text
                    print("Análisis Gemini OK.")
                    # Extraer Post_P del reporte generado
                    posterior_probability = extract_post_p(acmg_bayesian_report)
                    if posterior_probability is not None:
                        print(f"Post_P extraída: {posterior_probability}")
                else:
                    print("WARN: Gemini no devolvió contenido.")
                    acmg_bayesian_report = "El modelo IA no generó un reporte."
                    try: acmg_bayesian_report += f"\nFeedback: {response_combined.prompt_feedback}"
                    except Exception: pass
            else:
                 # Esto no debería ocurrir si VEP fue exitoso, pero por si acaso
                 pipeline_error_message = "Error interno: VEP exitoso pero sin datos para Gemini."
                 return {'error': True, 'message': pipeline_error_message, 'status_code': 500}

        except Exception as e:
            print(f"Error en Paso 3 (Gemini): {e}")
            # Devolver el error pero potencialmente con los datos VEP si se obtuvieron
            pipeline_error_message = f"Error durante el análisis con Gemini: {e}"
            # Podríamos devolver éxito parcial aquí si VEP funcionó, pero por simplicidad marcamos error
            return {'error': True, 'message': pipeline_error_message, 'status_code': 500}


        # --- Retornar resultados ---
        return {
            'error': False,
            'acmg_bayesian_report': acmg_bayesian_report,
            'posterior_probability': posterior_probability # Puede ser None si no se extrajo
        }

