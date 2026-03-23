-- stg_diabetes_admissions.sql
-- Staging model for UCI Diabetes 130-US Hospitals dataset
-- Responsibilities:
--   - Standardize column names (snake_case enforced)
--   - Convert '?' placeholders to NULL
--   - Cast columns to appropriate types
--   - Retain 'None' as meaningful category in lab test columns
--   - Create binary outcome variable (readmitted_30day)
--   - Pass all other columns through cleanly for marts transformation

WITH source AS (
    SELECT * FROM {{ source('raw', 'dat') }}
),

cleaned AS (
    SELECT
        -- Identifiers (retained for deduplication and ordering in marts)
        encounter_id::INT                                           AS encounter_id,
        patient_nbr::INT                                            AS patient_nbr,

        -- Demographics
        NULLIF(race, '?')                                           AS race,
        NULLIF(gender, 'Unknown/Invalid')                           AS gender,
        age                                                         AS age,
        NULLIF(weight, '?')                                         AS weight,

        -- Admission variables
        admission_type_id::INT                                      AS admission_type_id,
        discharge_disposition_id::INT                               AS discharge_disposition_id,
        admission_source_id::INT                                    AS admission_source_id,

        -- Hospital stay
        time_in_hospital::INT                                       AS time_in_hospital,
        num_lab_procedures::INT                                     AS num_lab_procedures,
        num_procedures::INT                                         AS num_procedures,
        num_medications::INT                                        AS num_medications,
        number_outpatient::INT                                      AS number_outpatient,
        number_emergency::INT                                       AS number_emergency,
        number_inpatient::INT                                       AS number_inpatient,
        number_diagnoses::INT                                       AS number_diagnoses,

        -- Clinical / administrative
        NULLIF(payer_code, '?')                                     AS payer_code,
        NULLIF(medical_specialty, '?')                              AS medical_specialty,

        -- Diagnosis codes (raw ICD-9 — grouping handled in marts)
        NULLIF(diag_1, '?')                                         AS diag_1,
        NULLIF(diag_2, '?')                                         AS diag_2,
        NULLIF(diag_3, '?')                                         AS diag_3,

        -- Lab results
        -- 'None' retained as meaningful category (test not administered)
        max_glu_serum                                               AS max_glu_serum,
        a1cresult                                                   AS a1cresult,

        -- Medications (all four-level categoricals: No/Steady/Up/Down)
        -- Near-zero variance assessment handled in marts
        metformin, repaglinide, nateglinide, chlorpropamide,
        glimepiride, acetohexamide, glipizide, glyburide,
        tolbutamide, pioglitazone, rosiglitazone, acarbose,
        miglitol, troglitazone, tolazamide, examide, citoglipton,
        insulin,
        glyburide_metformin,
        glipizide_metformin,
        glimepiride_pioglitazone,
        metformin_rosiglitazone,
        metformin_pioglitazone,

        -- Medication change flags
        change                                                      AS change,
        diabetesmed                                                 AS diabetes_med,

        -- Outcome variables
        readmitted                                                  AS readmitted_raw,

        CASE
            WHEN readmitted = '<30' THEN 1
            ELSE 0
        END                                                         AS readmitted_30day,

        CASE
            WHEN readmitted = 'NO' THEN 0
            ELSE 1
        END                                                         AS readmitted_any

    FROM source
)

SELECT * FROM cleaned