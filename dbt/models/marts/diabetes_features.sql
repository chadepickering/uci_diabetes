-- diabetes_features.sql
-- Marts model for UCI Diabetes 130-US Hospitals dataset
-- Responsibilities:
--   - Deduplicate to one encounter per patient (max time_in_hospital, then max encounter_id)
--   - Filter invalid discharge dispositions (deceased/hospice)
--   - Filter missing primary diagnosis
--   - Drop high-missingness and low-utility columns
--   - Engineer age midpoint and clinical age group
--   - Map ICD-9 codes to 9 disease categories
--   - Consolidate admission type into clinically meaningful groups
--   - Drop near-zero variance medication columns
--   - Engineer medication change summary features
--   - Engineer prior utilization composite
--   - Consolidate medical specialty
--   - Assign temporal split group based on encounter_id percentile

WITH staging AS (
    SELECT * FROM {{ ref('stg_diabetes_admissions') }}
),

-- Step 1: Deduplicate to one encounter per patient
-- Keep encounter with max time_in_hospital, break ties with max encounter_id
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY patient_nbr
            ORDER BY time_in_hospital DESC, encounter_id DESC
        ) AS rn
    FROM staging
),

deduped AS (
    SELECT * FROM ranked WHERE rn = 1
),

-- Step 2: Filter invalid discharge dispositions and missing primary diagnosis
-- Excluded discharge disposition codes:
--   11, 19, 20, 21 = Expired / Hospice (cannot be readmitted)
--   13, 14         = Hospice home / facility (cannot be readmitted)
--   12             = Still patient (discharge outcome ambiguous)
--   10             = Neonate transferred (clinically distinct population)
-- Excluded admission source codes:
--   11, 12, 13, 14, 23, 24 = Neonatal admissions (clinically distinct population)
-- Source: IDs_mapping.csv from UCI 130-US Hospitals dataset
filtered AS (
    SELECT * FROM deduped
    WHERE discharge_disposition_id NOT IN (11, 13, 14, 19, 20, 21, 12, 10)
    AND admission_source_id NOT IN (11, 12, 13, 14, 23, 24)
    AND diag_1 IS NOT NULL
),

-- Step 3: Feature engineering
featured AS (
    SELECT
        -- Outcome variables
        readmitted_raw,
        readmitted_30day,
        readmitted_any,

        -- Demographics
        race,
        gender,

        -- Age: raw band, numeric midpoint, and clinical grouping
        age                                                         AS age_band,
        CASE age
            WHEN '[0-10)'  THEN 5
            WHEN '[10-20)' THEN 15
            WHEN '[20-30)' THEN 25
            WHEN '[30-40)' THEN 35
            WHEN '[40-50)' THEN 45
            WHEN '[50-60)' THEN 55
            WHEN '[60-70)' THEN 65
            WHEN '[70-80)' THEN 75
            WHEN '[80-90)' THEN 85
            WHEN '[90-100)' THEN 95
        END                                                         AS age_midpoint,
        CASE
            WHEN age IN ('[0-10)', '[10-20)', '[20-30)')
                THEN 'young'
            WHEN age IN ('[30-40)', '[40-50)', '[50-60)')
                THEN 'middle'
            WHEN age IN ('[60-70)', '[70-80)', '[80-90)', '[90-100)')
                THEN 'senior'
        END                                                         AS age_group,

        -- Admission variables
        -- Raw IDs retained for reference
        admission_type_id,
        discharge_disposition_id,
        admission_source_id,

        -- Consolidated admission type
        CASE
            WHEN admission_type_id IN (1, 2, 7) THEN 'Emergency'
            WHEN admission_type_id = 3          THEN 'Elective'
            WHEN admission_type_id = 4          THEN 'Newborn'
            WHEN admission_type_id IN (5, 6, 8) THEN 'Unknown'
        END                                                         AS admission_type_group,

        -- Hospital stay variables (all retained)
        time_in_hospital,
        num_lab_procedures,
        num_procedures,
        num_medications,
        number_diagnoses,

        -- Prior utilization (source variables retained alongside composite)
        number_outpatient,
        number_emergency,
        number_inpatient,
        (number_outpatient + number_emergency + number_inpatient)   AS total_prior_encounters,

        -- Clinical variables
        -- payer_code dropped (high missingness)
        -- weight dropped (97% missing)
        medical_specialty,

        -- Medical specialty consolidation
        CASE
            WHEN medical_specialty ILIKE '%surgery%'          THEN 'Surgery'
            WHEN medical_specialty ILIKE '%internal%'         THEN 'Internal Medicine'
            WHEN medical_specialty = 'Cardiology'             THEN 'Cardiology'
            WHEN medical_specialty ILIKE '%emergency%'        THEN 'Emergency'
            WHEN medical_specialty IS NULL                    THEN 'Unknown'
            ELSE 'Other'
        END                                                         AS specialty_group,

        -- Diagnosis codes: raw ICD-9 retained alongside grouped versions
        diag_1,
        diag_2,
        diag_3,

        -- ICD-9 grouping function
        -- Applied to diag_1, diag_2, diag_3
        CASE
            WHEN diag_1 LIKE '250%'                               THEN 'Diabetes'
            WHEN TRY_CAST(diag_1 AS FLOAT) BETWEEN 390 AND 459
                OR diag_1 = '785'                                 THEN 'Circulatory'
            WHEN TRY_CAST(diag_1 AS FLOAT) BETWEEN 460 AND 519
                OR diag_1 = '786'                                 THEN 'Respiratory'
            WHEN TRY_CAST(diag_1 AS FLOAT) BETWEEN 520 AND 579
                OR diag_1 = '787'                                 THEN 'Digestive'
            WHEN TRY_CAST(diag_1 AS FLOAT) BETWEEN 580 AND 629
                OR diag_1 = '788'                                 THEN 'Genitourinary'
            WHEN TRY_CAST(diag_1 AS FLOAT) BETWEEN 800 AND 999   THEN 'Injury'
            WHEN TRY_CAST(diag_1 AS FLOAT) BETWEEN 710 AND 739   THEN 'Musculoskeletal'
            WHEN TRY_CAST(diag_1 AS FLOAT) BETWEEN 140 AND 239   THEN 'Neoplasms'
            ELSE 'Other'
        END                                                         AS diag_1_group,

        CASE
            WHEN diag_2 LIKE '250%'                               THEN 'Diabetes'
            WHEN TRY_CAST(diag_2 AS FLOAT) BETWEEN 390 AND 459
                OR diag_2 = '785'                                 THEN 'Circulatory'
            WHEN TRY_CAST(diag_2 AS FLOAT) BETWEEN 460 AND 519
                OR diag_2 = '786'                                 THEN 'Respiratory'
            WHEN TRY_CAST(diag_2 AS FLOAT) BETWEEN 520 AND 579
                OR diag_2 = '787'                                 THEN 'Digestive'
            WHEN TRY_CAST(diag_2 AS FLOAT) BETWEEN 580 AND 629
                OR diag_2 = '788'                                 THEN 'Genitourinary'
            WHEN TRY_CAST(diag_2 AS FLOAT) BETWEEN 800 AND 999   THEN 'Injury'
            WHEN TRY_CAST(diag_2 AS FLOAT) BETWEEN 710 AND 739   THEN 'Musculoskeletal'
            WHEN TRY_CAST(diag_2 AS FLOAT) BETWEEN 140 AND 239   THEN 'Neoplasms'
            ELSE 'Other'
        END                                                         AS diag_2_group,

        CASE
            WHEN diag_3 LIKE '250%'                               THEN 'Diabetes'
            WHEN TRY_CAST(diag_3 AS FLOAT) BETWEEN 390 AND 459
                OR diag_3 = '785'                                 THEN 'Circulatory'
            WHEN TRY_CAST(diag_3 AS FLOAT) BETWEEN 460 AND 519
                OR diag_3 = '786'                                 THEN 'Respiratory'
            WHEN TRY_CAST(diag_3 AS FLOAT) BETWEEN 520 AND 579
                OR diag_3 = '787'                                 THEN 'Digestive'
            WHEN TRY_CAST(diag_3 AS FLOAT) BETWEEN 580 AND 629
                OR diag_3 = '788'                                 THEN 'Genitourinary'
            WHEN TRY_CAST(diag_3 AS FLOAT) BETWEEN 800 AND 999   THEN 'Injury'
            WHEN TRY_CAST(diag_3 AS FLOAT) BETWEEN 710 AND 739   THEN 'Musculoskeletal'
            WHEN TRY_CAST(diag_3 AS FLOAT) BETWEEN 140 AND 239   THEN 'Neoplasms'
            ELSE 'Other'
        END                                                         AS diag_3_group,

        -- Lab results (None retained as meaningful category)
        max_glu_serum,
        a1cresult,

        -- Medication columns
        -- Near-zero variance columns dropped: examide, citoglipton,
        -- troglitazone, acetohexamide
        metformin,
        repaglinide,
        nateglinide,
        chlorpropamide,
        glimepiride,
        glipizide,
        glyburide,
        tolbutamide,
        pioglitazone,
        rosiglitazone,
        acarbose,
        miglitol,
        tolazamide,
        insulin,
        glyburide_metformin,
        glipizide_metformin,
        glimepiride_pioglitazone,
        metformin_rosiglitazone,
        metformin_pioglitazone,

        -- Medication change flags
        change,
        diabetes_med,

        -- Engineered medication summary features
        (
            CASE WHEN metformin              IN ('Up','Down') THEN 1 ELSE 0 END +
            CASE WHEN repaglinide            IN ('Up','Down') THEN 1 ELSE 0 END +
            CASE WHEN nateglinide            IN ('Up','Down') THEN 1 ELSE 0 END +
            CASE WHEN chlorpropamide         IN ('Up','Down') THEN 1 ELSE 0 END +
            CASE WHEN glimepiride            IN ('Up','Down') THEN 1 ELSE 0 END +
            CASE WHEN glipizide              IN ('Up','Down') THEN 1 ELSE 0 END +
            CASE WHEN glyburide              IN ('Up','Down') THEN 1 ELSE 0 END +
            CASE WHEN tolbutamide            IN ('Up','Down') THEN 1 ELSE 0 END +
            CASE WHEN pioglitazone           IN ('Up','Down') THEN 1 ELSE 0 END +
            CASE WHEN rosiglitazone          IN ('Up','Down') THEN 1 ELSE 0 END +
            CASE WHEN acarbose               IN ('Up','Down') THEN 1 ELSE 0 END +
            CASE WHEN miglitol               IN ('Up','Down') THEN 1 ELSE 0 END +
            CASE WHEN tolazamide             IN ('Up','Down') THEN 1 ELSE 0 END +
            CASE WHEN insulin                IN ('Up','Down') THEN 1 ELSE 0 END +
            CASE WHEN glyburide_metformin    IN ('Up','Down') THEN 1 ELSE 0 END +
            CASE WHEN glipizide_metformin    IN ('Up','Down') THEN 1 ELSE 0 END +
            CASE WHEN glimepiride_pioglitazone IN ('Up','Down') THEN 1 ELSE 0 END +
            CASE WHEN metformin_rosiglitazone IN ('Up','Down') THEN 1 ELSE 0 END +
            CASE WHEN metformin_pioglitazone IN ('Up','Down') THEN 1 ELSE 0 END
        )                                                           AS num_med_changes,

        (
            CASE WHEN metformin              != 'No' THEN 1 ELSE 0 END +
            CASE WHEN repaglinide            != 'No' THEN 1 ELSE 0 END +
            CASE WHEN nateglinide            != 'No' THEN 1 ELSE 0 END +
            CASE WHEN chlorpropamide         != 'No' THEN 1 ELSE 0 END +
            CASE WHEN glimepiride            != 'No' THEN 1 ELSE 0 END +
            CASE WHEN glipizide              != 'No' THEN 1 ELSE 0 END +
            CASE WHEN glyburide              != 'No' THEN 1 ELSE 0 END +
            CASE WHEN tolbutamide            != 'No' THEN 1 ELSE 0 END +
            CASE WHEN pioglitazone           != 'No' THEN 1 ELSE 0 END +
            CASE WHEN rosiglitazone          != 'No' THEN 1 ELSE 0 END +
            CASE WHEN acarbose               != 'No' THEN 1 ELSE 0 END +
            CASE WHEN miglitol               != 'No' THEN 1 ELSE 0 END +
            CASE WHEN tolazamide             != 'No' THEN 1 ELSE 0 END +
            CASE WHEN insulin                != 'No' THEN 1 ELSE 0 END +
            CASE WHEN glyburide_metformin    != 'No' THEN 1 ELSE 0 END +
            CASE WHEN glipizide_metformin    != 'No' THEN 1 ELSE 0 END +
            CASE WHEN glimepiride_pioglitazone != 'No' THEN 1 ELSE 0 END +
            CASE WHEN metformin_rosiglitazone != 'No' THEN 1 ELSE 0 END +
            CASE WHEN metformin_pioglitazone != 'No' THEN 1 ELSE 0 END
        )                                                           AS num_meds_active,

        -- Temporal split assignment based on encounter_id percentile
        CASE
            WHEN PERCENT_RANK() OVER (ORDER BY encounter_id) < 0.70
                THEN 'train'
            WHEN PERCENT_RANK() OVER (ORDER BY encounter_id) < 0.80
                THEN 'validation'
            ELSE 'holdout'
        END                                                         AS split_group

    FROM filtered
)

SELECT * FROM featured