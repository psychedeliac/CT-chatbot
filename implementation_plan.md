# Knowledge Base Enrichment Plan

## Background

The current RAG knowledge base has **237 document chunks** spread across **218 topics**. It covers general US consumer debt concepts well — FDCPA rights, debt collection rules, charge-offs, wage garnishments, credit cards, and debt settlement psychology.

**However, it has three critical blind spots for the Corporate Turnaround agent:**

1. **MCA (Merchant Cash Advance) is entirely absent** — no chunks explain what MCAs are, how their factor rates and daily remittance work, why they trap small businesses, or how to escape them.
2. **Business debt mechanics are thin** — the corpus focuses on consumer/personal debt. Business-specific topics like UCC liens, personal guarantees, vendor creditor hierarchy, and SBA loan defaults are missing.
3. **Corporate Turnaround's own process is underexplained** — the agent cannot explain *how* a turnaround engagement actually works step by step, what clients should expect, what fees look like, or how the program timeline unfolds.

---

## What We Need to Build

### Knowledge Pillar 1 — MCA (Merchant Cash Advance) Deep Dive
> The single highest-priority gap. Most CT clients are drowning in MCA stacks.

| Topic | What to Capture |
|:---|:---|
| What is an MCA? | Definition, how it works, distinction from a loan |
| Factor rates vs. interest rates | How 1.3–1.5 factor rates translate to 60–400% effective APR |
| Daily/weekly remittance | How ACH debits drain cash flow every business day |
| MCA stacking | How merchants take 2nd, 3rd advances; debt spiral mechanics |
| UCC-1 Financing Statements | How MCA funders file liens on all business assets |
| Confession of Judgment (COJ) | NY courts, no-notice judgments, bank account freezes |
| Personal guarantees in MCAs | How owners become personally liable |
| MCA default consequences | Acceleration clauses, bank account freezes, business closure |
| MCA vs. bank loan | Key structural differences — why MCAs are not regulated as loans |
| MCA negotiation | Can factor rates be reduced? Can daily pulls be paused? Settlement options |
| Legal landscape | State-level regulations (NY, CA), FTC scrutiny, pending legislation |

---

### Knowledge Pillar 2 — US Small Business Debt & Credit Mechanics
> Clients need foundational context on *how* business debt works vs. consumer debt.

| Topic | What to Capture |
|:---|:---|
| Secured vs. unsecured business debt | Definitions, lender hierarchy, collateral rules |
| Business credit scores | D&B PAYDEX, Equifax Business, Experian Business — how they differ from personal |
| SBA loan default | What happens when an SBA 7(a) or 504 loan defaults, OIC process |
| Business credit cards | Differing liability rules vs. personal cards, personal guarantee clauses |
| Vendor/trade credit | Net-30/60/90 terms, what happens when you default |
| Business lines of credit | How revolving credit facilities work, covenants |
| Revenue-based financing | Alternative to MCAs, how it compares |
| Business bankruptcy types | Chapter 7 (liquidation) vs Chapter 11 (reorganization) vs Chapter 13 — why CT avoids these |
| Out-of-court restructuring | How non-bankruptcy settlements work and why they're preferable |
| UCC Liens | How Article 9 filings work, how to search and release them |
| Personal guarantee | What it means for LLCs/S-corps, when owners become personally liable |
| Statute of limitations | Business debt SOL by state (differs from consumer SOL) |

---

### Knowledge Pillar 3 — Corporate Turnaround Process & Program
> The agent should be able to walk a prospect through the entire CT engagement.

| Topic | What to Capture |
|:---|:---|
| Free consultation process | What happens during the initial call, what info is needed |
| Client eligibility | Debt range ($15K–$30M), business types, states served |
| The CT program structure | Phase 1 (analysis), Phase 2 (negotiation), Phase 3 (settlement) |
| How CT negotiates | Creditor-by-creditor approach, leverage points, hardship letters |
| Client fee structure | How CT charges (% of enrolled debt), when fees are collected |
| What debts CT handles | Credit cards, MCAs, vendor debts, equipment leases, lawsuits |
| What debts CT cannot handle | Mortgages, car loans, taxes (IRS), child support |
| The SCORE partnership | Free business coaching available alongside CT program |
| IRS/tax specialist partners | What they handle that CT does not |
| Alternative lender partners | Cash flow financing during the program |
| Client timeline expectations | Typical program length (18–48 months), milestone checkpoints |
| Creditor harassment during program | What to say, cease-and-desist options, CT coaching on calls |
| Impact on credit score during program | Honest expectations, recovery timeline post-settlement |
| Success metrics | 18,000+ businesses, $800M+ settled, 50,000+ creditor settlements, BBB A+ |
| Alternatives to CT | Bankruptcy, self-negotiation, HELOC payoff — honest comparison |
| Industries CT serves | Trucking, restaurants, medical, construction, salons, gyms, etc. |

---

## Collection Strategy

All content will be collected via **automated scraping scripts** that fetch, parse, and format pages into the RAG chunk JSON schema. No manual copy-paste. Every source group gets its own dedicated scraper script.

### Scraper Architecture (shared pattern)

Each scraper will:
1. Accept a list of seed URLs (or crawl a sitemap)
2. Fetch the page HTML using `requests`
3. Parse with `BeautifulSoup4` to strip nav, ads, footers — keeping only article body text
4. Chunk text semantically by paragraph/heading into the target JSON schema
5. Write output to `data/raw/<source_group>.json`

```
scripts/
  scrapers/
    scrape_ct_site.py          # corporateturnaround.com
    scrape_gov_sources.py      # FTC, CFPB, SBA, SCORE, IRS
    scrape_industry_blogs.py   # Nav, Fundera, Investopedia, NerdWallet
    generate_qa_pairs.py       # Gemini-powered synthetic Q&A generation
  format_raw_to_chunks.py      # Normalise all raw -> unified chunk schema
```

---

### Source 1: corporateturnaround.com (`scrape_ct_site.py`)
**Method:** Sitemap crawl → full page scrape

Target pages to seed:
- `/about`, `/how-it-works`, `/faq`, `/services`, `/industries`, `/blog/*`, `/testimonials`

The scraper will:
- Crawl the sitemap or follow internal links from seed pages
- Extract `<h1>`, `<h2>`, `<p>` tags as section/chunk boundaries
- Auto-generate `id`, `topic`, `section_heading` from heading text
- Set `source_type: "corporate_turnaround_site"`

**Format Target:** Chunk into the same schema as `cleaned_blogs_rag.json`:
```json
{
  "id": "ct-about-000",
  "source_type": "corporate_turnaround_site",
  "topic": "...",
  "category": "...",
  "topic_tags": ["..."],
  "title": "...",
  "section_heading": "...",
  "chunk_text": "...",
  "chunk_tokens": 0,
  "requires_disclaimer": false
}
```

---

### Source 2: US Government & Regulatory Sources (`scrape_gov_sources.py`)
**Method:** Targeted URL list scrape (no login, all public domain pages)

| Agency | Seed URLs | Topics Covered |
|:---|:---|:---|
| FTC | ftc.gov/debt-collection, ftc.gov/mca | FDCPA, MCA enforcement actions |
| CFPB | consumerfinance.gov/small-business | MCA regulations, lending rules |
| SBA | sba.gov/funding-programs/loans | SBA loan default, OIC process |
| SCORE | score.org/resource | Business financial literacy |
| IRS | irs.gov/businesses/small | Tax debt context |

Set `source_type: "regulatory"`

---

### Source 3: Industry Blogs & Publications (`scrape_industry_blogs.py`)
**Method:** Targeted article URL list → article body extraction via `BeautifulSoup4`

| Publication | Topic Focus |
|:---|:---|
| nav.com/blog | MCA explained, UCC liens, business credit |
| fundera.com/blog | MCA stacking, factor rates vs APR, alternatives |
| investopedia.com | Chapter 7 vs 11, out-of-court restructuring, personal guarantees |
| nerdwallet.com/business | Business credit cards, SBA loans |
| score.org/blog | Cash flow, financial recovery |

Set `source_type: "educational"`

---

### Source 4: Synthetic Q&A Pairs (`generate_qa_pairs.py`)
**Method:** Script calls Gemini API with topic prompts to generate RAG-optimised Q&A pairs

The script will:
1. Take a predefined list of topic prompts (MCA, CT process, business debt)
2. Call `gemini-2.5-flash` to generate realistic prospect questions + grounded answers
3. Format output into the chunk schema with `source_type: "qa_pair"`

Target: ~150 Q&A pairs covering the top questions a CT prospect would ask.

Example prompt template used in the script:
```
Generate 10 realistic questions a small business owner might ask about Merchant Cash Advances,
along with clear, accurate answers from the perspective of a debt relief advisor.
Format as JSON with fields: question, answer.
```

Example output format:
```json
{
  "id": "qa-mca-001",
  "source_type": "qa_pair",
  "topic": "MCA Factor Rates",
  "category": "mca-education",
  "topic_tags": ["mca", "factor-rate", "cost-of-capital"],
  "title": "Understanding MCA Costs",
  "section_heading": "Q: What is a factor rate and how much does an MCA really cost?",
  "chunk_text": "A factor rate is a multiplier (typically 1.2 to 1.5) applied to the amount you borrow...",
  "chunk_tokens": 80,
  "requires_disclaimer": true
}
```

---

## Target Knowledge Base After Enrichment

| Pillar | Current Chunks | Target Chunks | Source Method |
|:---|:---:|:---:|:---|
| Consumer Debt (existing) | 237 | 237 | Keep as-is |
| MCA Deep Dive | 0 | ~80 | Scraped industry blogs + Q&A generation |
| US Business Debt Mechanics | ~10 | ~80 | Scraped gov sources + industry blogs |
| Corporate Turnaround Process | ~20 | ~100 | corporateturnaround.com scrape + Q&A generation |
| **Total** | **237** | **~497** | |

---

## Implementation Phases

### Phase 1 — Build Scrapers & Collect (Week 1)
- [ ] `scripts/scrapers/scrape_ct_site.py` — crawl corporateturnaround.com sitemap, extract all blog/FAQ/service/about pages
- [ ] `scripts/scrapers/scrape_gov_sources.py` — scrape FTC, CFPB, SBA, SCORE, IRS targeted public URLs
- [ ] `scripts/scrapers/scrape_industry_blogs.py` — scrape Nav, Fundera, Investopedia, NerdWallet articles on MCA/business debt topics
- [ ] `scripts/scrapers/generate_qa_pairs.py` — call Gemini API to generate ~150 synthetic Q&A pairs across all three knowledge pillars
- [ ] All raw output saved to `data/raw/` directory

### Phase 2 — Normalise & Merge (Week 1–2)
- [ ] `scripts/format_raw_to_chunks.py` — converts all `data/raw/*.json` files into the unified chunk schema
- [ ] Domain accuracy review of MCA-specific and legal content before ingestion (see Open Questions)
- [ ] Merge all chunk files into `data/enriched_knowledge_base.json`

### Phase 3 — New Loader & Re-ingest (Week 2)
- [ ] Create `data_handlers/enriched_loader.py` that reads `data/enriched_knowledge_base.json`
- [ ] Register it in `data_handlers/registry.py` as `"enriched"`
- [ ] Run `python scripts/ingest.py --loader enriched --force`
- [ ] Verify collection size and chunk count post-ingestion

### Phase 4 — Retrieval Quality Evaluation (Week 2)
- [ ] Define a test query suite of 30 representative prospect questions
- [ ] Run each query against the new collection, inspect retrieved chunks
- [ ] Score relevance; identify and fix poorly-covered topic areas
- [ ] Re-scrape or add Q&A pairs for weak spots

---

## Open Questions for Review

> [!WARNING]
> **MCA content requires accuracy.** Factor rates, COJ rules, and UCC-1 filing procedures differ by state. Any inaccurate scraped information here could mislead clients in a financially consequential way. All MCA content should be reviewed for domain accuracy before ingestion.

> [!IMPORTANT]
> **Verbatim scraping of third-party sites (Nav, Fundera, Investopedia)** may be a licensing concern. The Q&A generation script can be used to paraphrase content rather than store it verbatim. Government content (FTC, CFPB, SBA, IRS) is public domain and always safe.

> [!NOTE]
> **Synthetic Q&A pairs will have the highest retrieval quality** since they are semantically closest to how real users phrase questions. The `generate_qa_pairs.py` script should be run for every knowledge pillar, not just as a supplement.
