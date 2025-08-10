# /eval/build_labels.py
import json, random, re, datetime as dt, pathlib, yaml
OUT = pathlib.Path("data/audio"); OUT.mkdir(parents=True, exist_ok=True)

def normalize_money(s):
    s = s.lower().replace(',', '').strip()
    if 'lakh' in s: return int(float(re.findall(r'[\d\.]+', s)[0]) * 100000)
    m = re.findall(r'\d+', s); n = int(m[0]) if m else 0
    if 'k' in s: n *= 1000
    if 'hazaar' in s: n *= 1000
    return n

def resolve_date(token):
    today = dt.date.today()
    m = {'aaj':0, 'kal':1, 'parso':2, 'parsō':2}
    if token.lower() in m: return (today + dt.timedelta(days=m[token.lower()])).isoformat()
    # fallback next weekday names (somvaar..etc) left as exercise
    return today.isoformat()

def extract_slots(transcript):
    t = transcript.lower()
    slots = {}
    pc = re.search(r'\b(\d{6})\b', t)
    if pc: slots['pincode'] = pc.group(1)
    if 'day shift' in t: slots['preferred_shift'] = 'day'
    elif 'night shift' in t: slots['preferred_shift'] = 'night'
    if '2 wheeler hai' in t or 'two wheeler' in t or 'scooty hai' in t: slots['has_2wheeler']=True
    if 'nahi' in t and ('2 wheeler' in t or 'scooty' in t): slots['has_2wheeler']=False
    if 'english' in t: slots.setdefault('languages', []).append('en')
    if 'hindi' in t: slots.setdefault('languages', []).append('hi')
    sal = re.search(r'([0-9]{2}k|[0-9]+ hazaar|[0-9]{5,6}|beez hazaar|eighteen thousand|22k)', t)
    if sal:
        slots['expected_salary_inr_month'] = normalize_money(sal.group(1))
    exp = re.search(r'(\d+)\s*(saal|years?)', t)
    mon = re.search(r'(\d+)\s*(mahine|months?)', t)
    if exp: slots['total_experience_months'] = int(exp.group(1))*12
    if mon: slots['total_experience_months'] = int(mon.group(1))
    av = re.search(r'\b(aaj|kal|parso|parsō)\b', t)
    if av: slots['availability_date'] = resolve_date(av.group(1))
    return slots

def main():
    utter = yaml.safe_load(open("data/gen/utterances.yaml"))
    random.seed(7)
    items = []
    for i in range(1, 26):  # 25 clips
        # Make a “mini dialog line” that covers 2–3 slots each
        parts = [
            random.choice(utter['intents']['pincode']),
            random.choice(utter['intents']['shift']),
            random.choice(utter['intents']['salary']),
            random.choice(utter['intents']['twowheeler']),
            random.choice(utter['intents']['experience']),
            random.choice(utter['intents']['languages']),
            random.choice(utter['intents']['availability'])
        ]
        text = ". ".join(parts)
        slots = extract_slots(text)
        cid = f"clip{i:02d}"
        (OUT / cid).mkdir(parents=True, exist_ok=True)
        open(OUT / cid / "transcript.txt","w", encoding="utf-8").write(text)
        json.dump(slots, open(OUT / cid / "labels.json","w"), ensure_ascii=False, indent=2)
        items.append({"id":cid,"transcript":text,"slots":slots})
    json.dump(items, open("data/labels_index.json","w"), ensure_ascii=False, indent=2)
if __name__ == "__main__":
    main()
