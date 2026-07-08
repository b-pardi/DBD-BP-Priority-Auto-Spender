"""drives spender.run on a background thread for the run screen.

owns one spender.Switch (armed once, so the f7/f8 global hotkeys also work) and spawns/reuses a single
run thread. Start toggles idle<->running<->paused via the switch; Stop latches the kill and joins, and
a later Start resets the switch and spawns a fresh run (so the panic stop is real but recoverable
without re-registering hotkeys). stdout is teed into a queue for the run loop's [dry-run] log lines.
the screen reads state() to label its buttons; the app stops this on window close.
"""

import sys
import threading

from src import detect, ocr, spender


class _QueueWriter:
    """a minimal text stream that buffers writes into whole lines and pushes them onto a queue,
    so the run loop's print() output can be shown live in the ui log pane."""

    def __init__(self, q):
        self.q = q
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.q.put(line)

    def flush(self):
        pass


class RunController:
    def __init__(self, rows, log_queue, config, frame_sink=None):
        self.rows = rows
        self.log_queue = log_queue
        self.frame_sink = frame_sink  # debug screen's push_frame, or None
        # arm once: keys come from the config present at creation (rebinds need a restart to re-arm,
        # but the on-screen buttons always work).
        self.switch = spender.Switch(
            start_key=config.get("start_key", spender.START_KEY),
            kill_key=config.get("kill_key", spender.KILL_KEY),
        ).arm()
        self.thread = None

    def _build_source(self, config, sim):
        if sim:
            return spender.sim_source(self.rows, seed=0, low_conf_frac=0.2, discrepancy_frac=0.1)
        matcher = config.get("matcher", "cnn")
        ncc = detect.load_ncc_templates(self.rows) if matcher.startswith("ncc") else None
        # narrow the match library to the priority list's icons/sources, same as spender.main's cli
        # path; without this a ui-launched run compares every killer's add-ons on a survivor web
        # (the luckless-mouse-on-a-survivor-run bug). snapshotted at start, like the cli.
        row_pool = spender.build_pool_mask(
            self.rows, config.get("priorities", []),
            inferred=config.get("pool_inferred", True),
            exclusive=config.get("pool_exclusive", False),
        )
        if row_pool is not None:
            self.log_queue.put(
                f"pool: matching against {sum(row_pool)}/{len(self.rows)} library icons "
                f"({'priority-only' if config.get('pool_exclusive') else 'priority-inferred'})")
        return spender.live_source(
            self.rows, ncc, matcher=matcher,
            thresh_method=config.get("thresh_method", "adaptive_gaussian"),
            use_hough=config.get("node_finder", "contours") == "hough",
            auto_crop=config.get("auto_crop", True),
            web_bbox=config.get("web_bbox"),
            crop_pad_frac=config.get("crop_pad_frac", ocr.CROP_PAD_FRAC),
            debug=config.get("debug", False), row_pool=row_pool,
        )

    def start(self, config, sim, dry_run):
        """start a run (or pause/resume one already running).
        sim and dry_run force click off; only a live + non-dry run actually clicks in-game."""
        if self.thread and self.thread.is_alive():
            self.switch.toggle()  # pause <-> resume
            return
        self.switch.reset()  # clear a previous latched kill so the fresh run can proceed
        click = (not sim) and (not dry_run)
        source = self._build_source(config, sim)
        self.thread = threading.Thread(
            target=self._run, args=(source, config, click), daemon=True)
        self.thread.start()
        self.switch.toggle()  # idle -> running

    def _run(self, source, config, click):
        old = sys.stdout
        sys.stdout = _QueueWriter(self.log_queue)  # tee the loop's prints into the ui log
        try:
            spender.run(source, config, self.switch, self.rows, click=click,
                        debug=config.get("debug", False), frame_sink=self.frame_sink)
        except Exception as e:
            self.log_queue.put(f"run error: {type(e).__name__}: {e}")
        finally:
            sys.stdout = old
            self.log_queue.put("[run stopped]")

    def stop(self):
        """latched panic stop: kill the loop and join the thread (called by the Stop button and on
        window close)."""
        self.switch.kill()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def state(self):
        if self.switch.killed:
            return "Idle"
        if self.switch.running:
            return "Running"
        if self.thread and self.thread.is_alive():
            return "Paused"
        return "Idle"
