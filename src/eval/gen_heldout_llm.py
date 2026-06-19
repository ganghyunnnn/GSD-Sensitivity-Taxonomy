"""Independent (LLM-generated) held-out generalization test for the rule classifier
(removes author-bias + small-N concerns of test_classifier_heldout.py).
An LLM (gemma4:e4b) generates colloquial M1/M2/D queries with neutral prompts; the
rule classifier's recall on this set is the unbiased generalization measure.
Result (n=60): 55% M-recall (cf. 71% on the author-written set) -> reported as a
55--71% range across two independent held-out constructions."""
import io, json, re, sys
import requests
sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from taxonomy.classifier import TaxonomyClassifier

GEN = {
    "M1": "spatial-METRIC: asks a real-world distance/length/width/height/AREA in metres or square metres, or whether something is within X metres.",
    "M2": "CARDINALITY: asks to COUNT how many discrete objects, or a ratio/percentage of counts, or which side has more.",
    "D":  "DESCRIPTIVE: answerable by looking (scene type, presence/yes-no, colour, what it is, condition); NOT a measurement or count.",
}


def generate(model, kind, defn, n=20):
    prompt = (f"Write {n} short, casual, everyday ways a non-expert might ask a "
              f"remote-sensing image question that is a {kind} task.\n{defn}\n"
              "Vary wording a lot (slang, imperatives, indirect). Avoid the word 'GSD'.\n"
              "Output ONLY the questions, one per line, no numbering.")
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.9, "num_predict": 700}}, timeout=240)
    txt = r.json().get("response", "")
    qs = [re.sub(r'^[\d.\)\-\*\s"]+', '', l).strip().strip('"') for l in txt.splitlines() if len(l.strip()) > 8]
    return [q for q in qs if ('?' in q or len(q.split()) >= 4)][:n]


def main(model="gemma4:e4b"):
    clf = TaxonomyClassifier()
    allq = []
    for k, defn in GEN.items():
        for q in generate(model, k, defn):
            allq.append((q, k))
    res = {"M1": [0, 0], "M2": [0, 0], "D": [0, 0]}
    for q, exp in allq:
        r = clf.classify(q)
        pred = r.all_types[0].value if r.all_types else "D"
        res[exp][1] += 1
        res[exp][0] += (pred == exp)
    mh = res["M1"][0] + res["M2"][0]; mt = res["M1"][1] + res["M2"][1]
    print(f"LLM-generated held-out n={len(allq)} ({model}): M-recall {mh}/{mt}={mh/mt:.0%}; D-ok {res['D'][0]}/{res['D'][1]}")
    json.dump({"model": model, "n": len(allq), "m_recall": mh / mt if mt else 0, "by_type": res,
               "queries": [{"q": q, "exp": e} for q, e in allq]},
              open("experiments/heldout_llm.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
