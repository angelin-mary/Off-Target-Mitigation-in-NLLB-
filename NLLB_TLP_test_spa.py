import torch
from datasets import load_dataset
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sacrebleu.metrics import BLEU as SPMBLEU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.cuda.get_device_name(0))
translator = pipeline(
    "translation",
    model="/path/to/your/saved/model",  # Replace with your model path after finetuning
    device=0,
    num_beams=5,
    length_penalty=1.2,
    early_stopping=True,
)

id_model = AutoModelForSequenceClassification.from_pretrained(
    "/path/to/your/language_identifier_checkpoint").to(device)
id_tokenizer = AutoTokenizer.from_pretrained(
    "/path/to/your/language_identifier_checkpoint"
)

spbleu = SPMBLEU(tokenize="flores200")

ds = load_dataset("facebook/flores", "spa_latn", trust_remote_code=True)
src_sents = ds["devtest"]["sentence"]
N = len(src_sents)

targets = [
    "arb_Latn", "sat_Olck", "taq_Tfng", "min_Arab", "acm_Arab", "ars_Arab",
    "acq_Arab", "prs_Arab", "aka_Latn", "ary_Arab", "ajp_Arab", "dyu_Latn",
    "apc_Arab", "aeb_Arab", "arz_Arab", "kmb_Latn", "zho_Hant", "hrv_Latn",
    "awa_Deva", "bod_Tibt", "kin_Latn", "bjn_Arab", "knc_Arab", "ace_Arab",
    "arb_Arab"
]

off_count = {t: 0 for t in targets}
spbleu_scores = {}

label_map = {
    0: "ace_Arab", 1: "ace_Latn", 2: "acm_Arab", 3: "acq_Arab", 4: "aeb_Arab",
    5: "afr_Latn", 6: "ajp_Arab", 7: "aka_Latn", 8: "als_Latn", 9: "amh_Ethi",
    10: "apc_Arab", 11: "arb_Arab", 12: "arb_Latn", 13: "ars_Arab", 14: "ary_Arab",
    15: "arz_Arab", 16: "asm_Beng", 17: "ast_Latn", 18: "awa_Deva", 19: "ayr_Latn",
    20: "azb_Arab", 21: "azj_Latn", 22: "bak_Cyrl", 23: "bam_Latn", 24: "ban_Latn",
    25: "bel_Cyrl", 26: "bem_Latn", 27: "ben_Beng", 28: "bho_Deva", 29: "bjn_Arab",
    30: "bjn_Latn", 31: "bod_Tibt", 32: "bos_Latn", 33: "bug_Latn", 34: "bul_Cyrl",
    35: "cat_Latn", 36: "ceb_Latn", 37: "ces_Latn", 38: "cjk_Latn", 39: "ckb_Arab",
    40: "crh_Latn", 41: "cym_Latn", 42: "dan_Latn", 43: "deu_Latn", 44: "dik_Latn",
    45: "dyu_Latn", 46: "dzo_Tibt", 47: "ell_Grek", 48: "eng_latn", 49: "epo_Latn",
    50: "est_Latn", 51: "eus_Latn", 52: "ewe_Latn", 53: "fao_Latn", 54: "fij_Latn",
    55: "fin_Latn", 56: "fon_Latn", 57: "fra_Latn", 58: "fur_Latn", 59: "fuv_Latn",
    60: "gaz_Latn", 61: "gla_Latn", 62: "gle_Latn", 63: "glg_Latn", 64: "grn_Latn",
    65: "guj_Gujr", 66: "hat_Latn", 67: "hau_Latn", 68: "heb_Hebr", 69: "hin_Deva",
    70: "hne_Deva", 71: "hrv_Latn", 72: "hun_Latn", 73: "hye_Armn", 74: "ibo_Latn",
    75: "ilo_Latn", 76: "ind_Latn", 77: "isl_Latn", 78: "ita_Latn", 79: "jav_Latn",
    80: "jpn_Jpan", 81: "kab_Latn", 82: "kac_Latn", 83: "kam_Latn", 84: "kan_Knda",
    85: "kas_Arab", 86: "kas_Deva", 87: "kat_Geor", 88: "kaz_Cyrl", 89: "kbp_Latn",
    90: "kea_Latn", 91: "khk_Cyrl", 92: "khm_Khmr", 93: "kik_Latn", 94: "kin_Latn",
    95: "kir_Cyrl", 96: "kmb_Latn", 97: "kmr_Latn", 98: "knc_Arab", 99: "knc_Latn",
    100: "kon_Latn", 101: "kor_Hang", 102: "lao_Laoo", 103: "lij_Latn", 104: "lim_Latn",
    105: "lin_Latn", 106: "lit_Latn", 107: "lmo_Latn", 108: "ltg_Latn", 109: "ltz_Latn",
    110: "lua_Latn", 111: "lug_Latn", 112: "luo_Latn", 113: "lus_Latn", 114: "lvs_Latn",
    115: "mag_Deva", 116: "mai_Deva", 117: "mal_Mlym", 118: "mar_Deva", 119: "min_Arab",
    120: "min_Latn", 121: "mkd_Cyrl", 122: "mlt_Latn", 123: "mni_Beng", 124: "mos_Latn",
    125: "mri_Latn", 126: "mya_Mymr", 127: "nld_Latn", 128: "nno_Latn", 129: "nob_Latn",
    130: "npi_Deva", 131: "nso_Latn", 132: "nus_Latn", 133: "nya_Latn", 134: "oci_Latn",
    135: "ory_Orya", 136: "pag_Latn", 137: "pan_Guru", 138: "pap_Latn", 139: "pbt_Arab",
    140: "pes_Arab", 141: "plt_Latn", 142: "pol_Latn", 143: "por_Latn", 144: "prs_Arab",
    145: "quy_Latn", 146: "ron_Latn", 147: "run_Latn", 148: "rus_Cyrl", 149: "sag_Latn",
    150: "san_Deva", 151: "sat_Olck", 152: "scn_Latn", 153: "shn_Mymr", 154: "sin_Sinh",
    155: "slk_Latn", 156: "slv_Latn", 157: "smo_Latn", 158: "sna_Latn", 159: "snd_Arab",
    160: "som_Latn", 161: "sot_Latn", 162: "spa_Latn", 163: "srd_Latn", 164: "srp_Cyrl",
    165: "ssw_Latn", 166: "sun_Latn", 167: "swe_Latn", 168: "swh_Latn", 169: "szl_Latn",
    170: "tam_Taml", 171: "taq_Latn", 172: "taq_Tfng", 173: "tat_Cyrl", 174: "tel_Telu",
    175: "tgk_Cyrl", 176: "tgl_Latn", 177: "tha_Thai", 178: "tir_Ethi", 179: "tpi_Latn",
    180: "tsn_Latn", 181: "tso_Latn", 182: "tuk_Latn", 183: "tum_Latn", 184: "tur_Latn",
    185: "twi_Latn", 186: "tzm_Tfng", 187: "uig_Arab", 188: "ukr_Cyrl", 189: "umb_Latn",
    190: "urd_Arab", 191: "uzn_Latn", 192: "vec_Latn", 193: "vie_Latn", 194: "war_Latn",
    195: "wol_Latn", 196: "xho_Latn", 197: "ydd_Hebr", 198: "yor_Latn", 199: "yue_Hant",
    200: "zho_Hans", 201: "zho_Hant", 202: "zsm_Latn", 203: "zul_Latn"
}

def predict_language(text):
    inp = id_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
    out = id_model(**inp)
    return label_map[torch.argmax(out.logits, dim=1).item()]

for tgt in targets:
    if tgt == "spa_latn":
        continue

    print(f"\n→ spa_latn → {tgt}")
    ref_ds = load_dataset("facebook/flores", tgt, trust_remote_code=True)
    refs = ref_ds["devtest"]["sentence"][:N]

    good_refs, good_preds = [], []

    for i, src in enumerate(src_sents):
        try:
            out = translator(src, src_lang="spa_latn", tgt_lang=tgt, max_length=400)
            pred = out[0]["translation_text"]
            lang = predict_language(pred)
            off = (lang != tgt)
            if off:
                off_count[tgt] += 1
            print(f"[{i+1}/{N}] off={off} → {pred}")

            if not off:
                good_refs.append(refs[i])
                good_preds.append(pred)
        except:
            pass

    if good_preds:
        score = spbleu.corpus_score(good_preds, [good_refs]).score / 100
        spbleu_scores[tgt] = score
        print(f"spBLEU: {score:.4f}")
    else:
        print("spBLEU: n/a")

print("\nOff-target rates:")
for t, c in off_count.items():
    print(f"  {t}: {c/N:.2%}")

print("\nspBLEU scores:")
tot = 0
for t, s in spbleu_scores.items():
    print(f"  {t}: {s:.4f}")
    tot += s
print(f"\nAverage spBLEU: {tot/len(spbleu_scores):.4f}")
