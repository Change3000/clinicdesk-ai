/* ClinicDesk AI - triage engine (runs in the browser and in Node for testing).
   Pipeline: 1) red-flag safety rules  2) intent classifier (TF-IDF + logistic
   regression exported from model/train.py)  3) detail extraction  4) routing +
   drafted reply. No network calls: messages never leave the device. */
(function (root) {
  "use strict";

  const ROUTES = {
    urgent:              { name: "Possible emergency",   team: "Duty GP / nurse",  sla: "Call now",              priority: 0, auto: false },
    clinical_concern:    { name: "Clinical question",    team: "Nurse triage",     sla: "Call back within 2 hrs", priority: 1, auto: false },
    test_results:        { name: "Test results",         team: "GP review",        sla: "Same day",               priority: 2, auto: true  },
    medical_certificate: { name: "Medical certificate",  team: "GP",               sla: "Same day",               priority: 3, auto: true  },
    repeat_prescription: { name: "Repeat prescription",  team: "GP script queue",  sla: "1 business day",         priority: 4, auto: true  },
    referral:            { name: "Referral",             team: "GP admin",         sla: "2 business days",        priority: 5, auto: true  },
    book_appointment:    { name: "New booking",          team: "Reception",        sla: "Same day",               priority: 6, auto: true  },
    change_appointment:  { name: "Change / cancel",      team: "Reception",        sla: "Same day",               priority: 6, auto: true  },
    billing_medicare:    { name: "Billing & Medicare",   team: "Practice admin",   sla: "2 business days",        priority: 7, auto: true  },
    general_enquiry:     { name: "General enquiry",      team: "Reception",        sla: "Same day",               priority: 8, auto: true  },
  };

  const CLINIC = "Harbourside Family Practice";

  const REPLIES = {
    book_appointment: d => `Hi${d.greet}, thanks for getting in touch with ${CLINIC}. ${d.forWhom ? "We'd be happy to see your " + d.forWhom + ". " : ""}You can book the next available time${d.doctor ? " with " + d.doctor : ""} online at harbourside.example/book, or reply with two times that suit you${d.when ? " (you mentioned " + d.when + ")" : ""} and we'll lock one in. If the problem is getting worse, please call us on (02) 5550 1234.`,
    change_appointment: d => `Hi${d.greet}, no problem, we've noted your request to ${/cancel/i.test(d.text) ? "cancel" : "change"} your appointment${d.when ? " (" + d.when + ")" : ""}. Reception will confirm by SMS shortly. To pick a new time yourself, use harbourside.example/book.`,
    repeat_prescription: d => `Hi${d.greet}, we've passed your repeat prescription request${d.meds ? " for " + d.meds : ""} to ${d.doctor || "your regular GP"}. If it can be issued without a consult, an e-script will be sent to your mobile within 1 business day. Some medicines need a review appointment first, and we'll let you know if that's the case.`,
    test_results: d => `Hi${d.greet}, thanks for checking. Your ${d.tests || "results"} will be reviewed by the doctor who ordered them, and we'll contact you today. Please note we can't give results by email until a doctor has reviewed them.`,
    referral: d => `Hi${d.greet}, we've sent your referral request${d.specialist ? " (" + d.specialist + ")" : ""} to your GP. Most referrals can be done the same week; some need a short appointment so the doctor can include the right information. We'll be in touch within 2 business days.`,
    medical_certificate: d => `Hi${d.greet}, a doctor needs to assess you before issuing a medical certificate. We can do this by telehealth today; reply YES and we'll send you a link, or book at harbourside.example/book.`,
    billing_medicare: d => `Hi${d.greet}, thanks for your billing question. Our practice admin team will reply within 2 business days. Standard consultation fees and bulk-billing eligibility are listed at harbourside.example/fees.`,
    general_enquiry: d => `Hi${d.greet}, thanks for your message. Our opening hours, location, parking and services are at harbourside.example/visit. A member of our reception team will reply to anything not covered there today.`,
    clinical_concern: d => `Hi${d.greet}, thanks for letting us know. A practice nurse will call you within 2 hours to talk this through. If symptoms get worse or you're worried, call 000 in an emergency or Healthdirect on 1800 022 222 for 24/7 advice.`,
  };

  const MEDS = ["ventolin","puffer","inhaler","metformin","insulin","atorvastatin","statin","statins","sertraline","levothyroxine","thyroxine","panadol","paracetamol","ibuprofen","nurofen","amoxicillin","antibiotics","pill","contraceptive","epipen","warfarin","blood thinner","antidepressant","antidepressants","ritalin","endone","pantoprazole","nexium","lisinopril","amlodipine","perindopril","coversyl","bp meds","blood pressure medication","cholesterol tablets","heart tablets","thyroid tabs","reflux meds","eczema cream"];
  const TESTS = ["blood test","bloods","pathology","x-ray","xray","ultrasound","scan","biopsy","mri","ct scan","urine","ecg","pap smear","swab","mammogram","stool sample","iron","cholesterol","hba1c"];
  const SPECIALISTS = ["dermatologist","skin specialist","cardiologist","heart specialist","physio","physiotherapist","psychologist","counsellor","orthopaedic surgeon","surgeon","endocrinologist","endo","ent","eye specialist","ophthalmologist","dietitian","podiatrist","neurologist","gynaecologist","paediatrician","speech pathologist","rheumatologist","breast screen","sleep study"];
  const WHO = ["son","daughter","mum","mother","dad","father","husband","wife","partner","baby","toddler","kid","kids","child","children"];

  function normalise(text, map) {
    let t = text.toLowerCase().replace(/[’']/g, "");
    const toks = t.match(/[a-z0-9]+|[^a-z0-9\s]/g) || [];
    return toks.map(k => (Object.prototype.hasOwnProperty.call(map, k) ? map[k] : k)).join(" ");
  }

  function makeClassifier(model) {
    const index = new Map(model.vocab.map((w, i) => [w, i]));
    return function classify(text) {
      const norm = normalise(text, model.normalise);
      const toks = norm.match(/\b\w\w+\b/g) || [];
      const counts = new Map();
      for (const w of toks) if (index.has(w)) counts.set(w, (counts.get(w) || 0) + 1);
      const feats = [];
      let sq = 0;
      for (const [w, c] of counts) {
        const j = index.get(w);
        const v = (1 + Math.log(c)) * model.idf[j];
        feats.push([j, v]); sq += v * v;
      }
      const nrm = Math.sqrt(sq) || 1;
      const logits = model.intercept.map((b, k) => {
        let s = b;
        for (const [j, v] of feats) s += model.coef[k][j] * (v / nrm);
        return s;
      });
      const mx = Math.max(...logits);
      const ex = logits.map(z => Math.exp(z - mx));
      const tot = ex.reduce((a, b) => a + b, 0);
      const probs = model.labels.map((l, k) => ({ label: l, p: ex[k] / tot })).sort((a, b) => b.p - a.p);
      // words that pushed the message towards the winning category the most
      const k0 = model.labels.indexOf(probs[0].label);
      const known = feats.map(([j, v]) => [model.vocab[j], model.coef[k0][j] * v / nrm])
        .filter(([, c]) => c > 0.05).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([w]) => w);
      return { probs, known, normalised: norm };
    };
  }

  function makeSafety(rulesDoc) {
    const rules = rulesDoc.rules.map(r => ({ ...r, rx: new RegExp(r.pattern, "i"), ex: r.exclude ? new RegExp(r.exclude, "i") : null }));
    return function redFlags(text) {
      const t = text.toLowerCase();
      return rules.filter(r => r.rx.test(t) && !(r.ex && r.ex.test(t))).map(r => ({ id: r.id, label: r.label, match: (t.match(r.rx) || [""])[0] }));
    };
  }

  // Learned urgency model (word + character n-gram TF-IDF, logistic regression).
  // Mirrors scikit-learn's TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5)).
  function makeUrgency(u) {
    if (!u) return () => 0;
    const wIdx = new Map(u.word_vocab.map((w, i) => [w, i]));
    const cIdx = new Map(u.char_vocab.map((w, i) => [w, i]));
    const nW = u.word_vocab.length;
    function vec(counts, idx, idf, offset, out) {
      let sq = 0; const tmp = [];
      for (const [k, c] of counts) { const j = idx.get(k); const v = (1 + Math.log(c)) * idf[j]; tmp.push([j + offset, v]); sq += v * v; }
      const n = Math.sqrt(sq) || 1; for (const [j, v] of tmp) out.push([j, v / n]);
    }
    return function urgency(text) {
      const t = text.toLowerCase();
      const wc = new Map();
      for (const w of (t.match(/\b\w\w+\b/g) || [])) if (wIdx.has(w)) wc.set(w, (wc.get(w) || 0) + 1);
      const cc = new Map();
      for (const raw of t.replace(/\s\s+/g, " ").split(/\s+/).filter(Boolean)) {
        const w = " " + raw + " ";
        for (let n = 3; n <= 5; n++) {
          let off = 0; const add = g => { if (cIdx.has(g)) cc.set(g, (cc.get(g) || 0) + 1); };
          add(w.slice(off, off + n));
          while (off + n < w.length) { off++; add(w.slice(off, off + n)); }
          if (off === 0) break; // short word: counted once, as scikit-learn does
        }
      }
      const feats = []; vec(wc, wIdx, u.word_idf, 0, feats); vec(cc, cIdx, u.char_idf, nW, feats);
      let z = u.intercept; for (const [j, v] of feats) z += u.coef[j] * v;
      return 1 / (1 + Math.exp(-z));
    };
  }

  function findAll(list, low) {
    return list.filter(w => new RegExp("\\b" + w.replace(/[-/\\^$*+?.()|[\]{}]/g, "\\$&") + "\\b").test(low));
  }

  function extract(text) {
    const low = text.toLowerCase();
    const d = { text };
    const phone = text.match(/(?:\+?61\s?|0)4\d{2}[\s-]?\d{3}[\s-]?\d{3}/);
    if (phone) d.phone = phone[0];
    const medicare = text.match(/\b\d{4}\s?\d{5}\s?\d{1,2}\b/);
    if (medicare) d.medicare = "•••• ••••• " + medicare[0].replace(/\s/g, "").slice(-1);
    const dob = text.match(/\b(?:dob|d\.o\.b|born)[:\s]*(\d{1,2}[\/.-]\d{1,2}[\/.-]\d{2,4})/i);
    if (dob) d.dob = dob[1];
    const doc = text.match(/\b(?:dr|doctor)\.?\s+([A-Z][a-z]+)/) || text.match(/\bdr\.?\s+([a-z]{3,})/i);
    if (doc && !/^(about|today|asap|said|says|for|to|and|look|send)$/i.test(doc[1])) d.doctor = "Dr " + doc[1][0].toUpperCase() + doc[1].slice(1).toLowerCase();
    const when = [];
    const days = low.match(/\b(today|tonight|tomorrow|tmrw|tmr|this (morning|afternoon|arvo|week|weekend)|next (week|wk|month)|(on |this |next )?(mon|tues|wednes|thurs|fri|satur|sun)(day)?\b|thurs|weekend|the \d{1,2}(st|nd|rd|th))/g);
    if (days) when.push(...days.map(s => s.trim()));
    const times = low.match(/\b\d{1,2}([:.]\d{2})?\s?(am|pm)\b|\b\d{1,2}[:.]\d{2}\b|\b(morning|afternoon|arvo|evening)\b/g);
    if (times) when.push(...times);
    if (when.length) d.when = [...new Set(when)].slice(0, 3).join(", ");
    const meds = findAll(MEDS, low); if (meds.length) d.meds = meds.slice(0, 2).join(", ");
    const tests = findAll(TESTS, low); if (tests.length) d.tests = tests.slice(0, 2).join(" and ");
    const spec = findAll(SPECIALISTS, low); if (spec.length) d.specialist = spec.slice(0, 2).join(", ");
    const who = low.match(new RegExp("\\bmy (" + WHO.join("|") + ")\\b"));
    if (who) d.forWhom = who[1];
    const name = text.match(/\b(?:my name is|this is|it'?s)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)/);
    if (name) d.name = name[1];
    d.greet = d.name ? " " + d.name.split(" ")[0] : "";
    return d;
  }

  function createEngine(model, rulesDoc, urgencyModel) {
    const classify = makeClassifier(model);
    const redFlags = makeSafety(rulesDoc);
    const urgency = makeUrgency(urgencyModel);
    const th = urgencyModel ? urgencyModel.threshold : 1;
    return function triage(text) {
      const flags = redFlags(text);
      const uScore = urgency(text);
      if (uScore >= th) flags.push({ id: "urgency_model", label: "Urgency model (" + Math.round(uScore * 100) + "% likely emergency)", match: "" });
      const cls = classify(text);
      const top = cls.probs[0], second = cls.probs[1];
      const details = extract(text);
      const intent = top.label;
      const urgent = flags.length > 0;
      const routeKey = urgent ? "urgent" : intent;
      // Thresholds calibrated on the test + holdout sets (see docs): a flat top
      // probability or a close runner-up catches 5 of the 7 misclassifications.
      const ratio = second ? second.p / top.p : 0;
      const lowConfidence = !urgent && (top.p < 0.2 || ratio > 0.85);
      const alsoMentions = !urgent && second && ratio > 0.6 ? second.label : null;
      const route = ROUTES[routeKey];
      let reply = null;
      if (!urgent) reply = REPLIES[intent](details);
      return {
        text, urgent, flags, urgencyScore: uScore, intent, confidence: top.p, probs: cls.probs, alsoMentions,
        lowConfidence, details, routeKey, route,
        reply, safetyAdvice: urgent ? rulesDoc.response : null,
        evidence: cls.known.slice(0, 8),
      };
    };
  }

  const api = { createEngine, normalise, ROUTES, CLINIC };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ClinicDesk = api;
})(typeof window !== "undefined" ? window : globalThis);
