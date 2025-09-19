<!-- Repository-level Copilot instructions for `notes-deep-learning`. -->

Goal: Help AI agents produce correct, idiomatic, and repo-consistent code for experiments,
notebooks, and small scripts. Focus on reproducibility and readable, well-documented examples.

How this repo is organized (big picture)

- `notebooks/` — annotated Jupyter notebooks used for teaching and experiments. Prefer small
  incremental code changes and keep outputs minimal when editing.
- `examples/` — runnable .py and .ipynb examples. These are canonical; mirror patterns from
  `examples/04_air_passengers_dataset.py` when producing runnable scripts.
- `Deep_Learning_Specialization/` — course materials and PDFs (read-only). Do not modify.
- `Text_Analytics_AUEB/` — course materials, notebooks and PDFs (read-only). Do not modify. Ignore.
- `PyTorch_Youtube_Luke_Ditria/` — course materials, notebooks and PDFs (read-only). Do not modify. Ignore.

Patterns & conventions specific to this repo

- Notebooks and scripts intentionally favour clarity over micro-optimizations. Prefer explicit
  loops and well-named intermediate variables for pedagogical clarity.

What to do when editing or generating code

- Keep changes small and focused. For notebooks, update a single cell at a time and include a short
  markdown explanation of why the change was made.
- Add or update a one-line docstring for any new public function.

Notebook-specific guidelines

- **Stay focused**: When generating `.ipynb` files, do exactly what the user asked. Only add
  improvements or innovations if they are highly relevant and you are absolutely certain they're
  within scope. Even then, keep additions minimal.
- **Structure with markdown**: Split notebook content into logical sections using markdown headers
  (`# Section`, `## Subsection`). Avoid creating overly long or extended cells.
- **Concise cells**: PREFER MULTIPLE FOCUSES CELLS over single sprawling ones for better readability
  and teaching flow.

Function standards

- **Always add docstrings**: Use Google-style docstrings for all functions. Keep them concise but informative.
- **Type annotations**: Always include proper type annotations for function parameters and return values.
- **When reviewing scripts**: Ensure existing functions follow the above standards—add missing docstrings and type annotations.

If something is unclear

- Ask for exact intent (e.g., "refactor example X to use dataset Y", or "convert notebook Z to script").
