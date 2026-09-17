# tau² Trace Viewer

A dependency-free, read-only local browser for saved tau² simulations. It discovers
`results.json` files below `data/simulations` and presents the available run names in
a dropdown—there is nothing to upload manually.

## Run it

From the base `tau2-bench` directory:

```bash
uv run python tools/tau2-viewer/server.py
```

The viewer opens at <http://127.0.0.1:4173>. Choose a run from the dropdown. If a
new tau² simulation finishes while the page is open, click **Refresh runs**.

The page shows run-level metrics, simulation outcomes, model configuration,
messages, tool calls, tool results, token counts, runtime, cost, and the raw
simulation record.

## Options

To use another simulations directory or port:

```bash
uv run python tools/tau2-viewer/server.py \
  --simulations-dir /path/to/data/simulations \
  --port 4174
```

Use `--no-browser` to skip opening a browser automatically. Use `--check` to print
the number of discovered runs and exit.

## Data handling

- The server binds to `127.0.0.1` by default and exposes read-only GET endpoints.
- Trace data remains on the local computer and is not uploaded or modified.
- Standard inline simulations and voice runs split into a `simulations/` directory
  are supported.
- The viewer does not read `.env` files.
