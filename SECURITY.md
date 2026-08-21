# Security Policy

We take the security of Mesopic seriously. Mesopic is privacy-by-design — video
never leaves the customer's premises — but the cloud sync channel, the hosted
dashboard, and the billing path are real attack surface, and we welcome reports.

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report privately via either:

- **GitHub Private Vulnerability Reporting** — the Security tab → "Report a
  vulnerability". Preferred: it keeps the report, the fix, and the advisory in one
  place, and it credits you automatically if you want that.
- **Email: mesopic.project@gmail.com** — if you would rather not use GitHub, or your
  report does not fit an advisory form.

Include: a description, steps to reproduce, the affected component and version, and the
impact you believe it has. A minimal proof-of-concept helps enormously.

## Scope

**In scope:**
- The engine in this repository, and the container images published from it.
- Anything that would let an attacker get video, frames, or images off a box running
  Mesopic, or persist them to disk. That is the invariant the whole product rests on, and
  a working break of it is the highest-value report you can send us.
- Credential handling: RTSP URLs, webhook secrets, and the cloud site token.

The hosted cloud service is **not yet deployed**; when it launches, its scope and
endpoints will be listed here.

**Out of scope:**
- The customer's own cameras, LAN, and RTSP credentials — camera/network
  security is the operator's responsibility (see our threat model).
- Denial-of-service via traffic volume, social engineering, physical attacks, and
  findings in third-party services — report those to the vendor.
- Automated-scanner output without a demonstrated, reproducible impact.

## Safe harbour

We will not pursue legal action against, or ask law enforcement to investigate,
researchers who:
- make a good-faith effort to comply with this policy,
- test only against **their own** installation and data, and do not access, modify, or
  exfiltrate anyone else's,
- do not degrade the service for others, and
- give us reasonable time to remediate before any public disclosure.

Activity conducted consistently with this policy is considered authorised, and we
will not consider it a violation of applicable computer-misuse law.

## Our commitment (expected response)

Mesopic is early and maintained by a very small team; these timelines are set to be
achievable rather than impressive:

| Stage | Target |
|---|---|
| Acknowledge your report | within 3 business days |
| Initial assessment + severity | within 7 business days |
| Fix or mitigation for High/Critical | within 30 days (faster where feasible) |
| Public disclosure / credit | coordinated with you after a fix ships |

There is **no paid bug bounty**, and we would rather say that plainly than imply one.
We will publicly credit reporters who want it. This policy will be revised as the project
and the team grow.
