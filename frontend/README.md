# frontend

The browser surface for the agent. Adapted from a market-dashboard repo, so the visual language is
inherited; what it shows is not. Every page here is backed by an endpoint `backend/main.py` actually
serves, and the reference repo's pages that called endpoints this project has no answer for were
removed rather than shipped returning errors.

- `/` — ask the agent. Opens the `WS /chat` socket and renders each stage as it arrives. Shows the
  answer, the evidence it was given, the routing margin and confidence, and the verdict once the turn
  is stored. When the agent's confidence sits under the cut it fitted from past feedback, the answer
  is withheld and the page says so; that threshold is the only thing in the system that learns.
- `/portfolio` — simulated positions folded from the trade log at average cost. No money moves.
- `/login` — register and log in against `POST /auth/register|login`. The token lives in
  `localStorage`; there is no third-party identity provider.

## Running it

The backend must already be up, because nothing here holds a copy of the model:

```
python3 scripts/serve.py      # from the repository root, serves 127.0.0.1:8000
npm install && npm run dev    # from here, serves localhost:3000
```

Point it elsewhere with `NEXT_PUBLIC_API_URL`.

## The gate

```
node verify_gate.mjs
```

Drives a real Chromium against both servers: registers an account through the form, types questions
into the page, and asserts each reply rendered its answer and the evidence behind it, with every
figure in a spoken answer present in that evidence. It types rather than posts on purpose — anything
reachable by an HTTP request is already covered by `tests/`, so the only thing left worth measuring
here is what a request cannot see. It earned that: it caught a promise-returning `scrollIntoView`
being taken for a React cleanup function, which tore the page down on the first socket update.
