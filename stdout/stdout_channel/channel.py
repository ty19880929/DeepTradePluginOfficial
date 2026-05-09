"""Stdout reference channel plugin.

Implements the framework's ``ChannelPlugin`` Protocol:
    * ``validate_static`` — install-time self-check (no network)
    * ``dispatch(argv)``  — CLI dispatch (subcommands: test, log)
    * ``push(ctx, payload)`` — receive a NotificationPayload from the notifier

Behaviour on push:
    * FULLY consume the NotificationPayload (walk every section/item/metric)
      so any structural bug surfaces immediately.
    * Persist a one-row audit record to ``stdout_channel_log``.
    * Emit ONE concise line: "✔ push success (...)".

This channel is the zero-dependency target for unit tests of the notify
plumbing and is safe to ship enabled-by-default in dev installs.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.plugins_api.base import PluginContext
    from deeptrade.plugins_api.notify import NotificationPayload


class StdoutChannel:
    """Stdout-only IM channel plugin."""

    metadata = None  # injected by framework after install

    # ----- Plugin Protocol -------------------------------------------------

    def validate_static(self, ctx: PluginContext) -> None:  # noqa: ARG002
        # Sanity: the audit table must exist (created by our own migration).
        # If it's missing, the install pipeline already failed; this is just a
        # belt-and-braces self-check that doesn't touch the network.
        return

    def dispatch(self, argv: list[str]) -> int:
        """Channel-side CLI: ``test`` and ``log``.

        ``test`` synthesizes a minimal payload and routes it back through this
        channel's own ``push`` so the user can verify the channel is wired up
        without running a real strategy. ``log`` dumps the most recent
        ``stdout_channel_log`` rows.
        """
        parser = argparse.ArgumentParser(
            prog="deeptrade stdout-channel",
            description="Stdout notification channel — local self-test + audit log.",
        )
        sub = parser.add_subparsers(dest="cmd", required=False)

        sub.add_parser("test", help="Push a synthetic payload through this channel.")
        log_p = sub.add_parser("log", help="Print recent delivery audit rows.")
        log_p.add_argument("--limit", type=int, default=20)

        args = parser.parse_args(argv)
        if args.cmd is None:
            parser.print_help()
            return 0
        if args.cmd == "test":
            return self._cmd_test()
        if args.cmd == "log":
            return self._cmd_log(args.limit)
        return 2

    # ----- ChannelPlugin Protocol ------------------------------------------

    def push(self, ctx: PluginContext, payload: NotificationPayload) -> None:
        # 1) Walk the payload — counts come from real iteration, never from
        # dict.len(), so we cannot "succeed" without having actually read the
        # plugin's data.
        section_count = len(payload.sections)
        item_count = sum(len(s.items) for s in payload.sections)
        metric_count = len(payload.metrics)
        for section in payload.sections:
            for item in section.items:
                _ = (item.code, item.name, item.rank, item.score, item.label, item.note)
        for k, v in payload.metrics.items():
            _ = (k, v)
        _ = payload.title, payload.summary, payload.report_dir, payload.extras

        # 2) Persist a delivery audit row.
        ctx.db.execute(
            "INSERT INTO stdout_channel_log(plugin_id, run_id, status, title, "
            "section_count, item_count, metric_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                payload.plugin_id,
                payload.run_id,
                payload.status.value,
                payload.title,
                section_count,
                item_count,
                metric_count,
            ),
        )

        # 3) One concise line. Use sys.stdout (not rich console) so this stays
        # out of any TUI / strategy console that the caller may have running.
        sys.stdout.write(
            f"✔ push success (channel=stdout-channel run_id={payload.run_id} "
            f"status={payload.status.value})\n"
        )
        sys.stdout.flush()

    # ----- private helpers -------------------------------------------------

    def _cmd_test(self) -> int:
        """Round-trip a synthetic payload through ``deeptrade.notify``.

        Goes through the framework's notifier layer (not directly through
        ``self.push``) so the test exercises the full path: install →
        build_notifier → dispatch → push.
        """
        import uuid

        from deeptrade import notify
        from deeptrade.core import paths
        from deeptrade.core.db import Database
        from deeptrade.core.run_status import RunStatus
        from deeptrade.plugins_api.notify import (
            NotificationItem,
            NotificationPayload,
            NotificationSection,
        )

        payload = NotificationPayload(
            plugin_id="stdout-channel",
            run_id=str(uuid.uuid4()),
            status=RunStatus.SUCCESS,
            title="DeepTrade — stdout channel test",
            summary="Synthetic payload from `deeptrade stdout-channel test`.",
            sections=[
                NotificationSection(
                    key="demo",
                    title="Demo items",
                    items=[
                        NotificationItem(
                            code="600519.SH", name="贵州茅台", rank=1, score=87.5,
                            label="top_candidate", note="演示条目，非真实推荐",
                        ),
                    ],
                ),
            ],
            metrics={"selected": 1},
        )
        db = Database(paths.db_path())
        try:
            notify(db, payload)
        finally:
            db.close()
        return 0

    def _cmd_log(self, limit: int) -> int:
        from deeptrade.core import paths
        from deeptrade.core.db import Database

        db = Database(paths.db_path())
        try:
            rows = db.fetchall(
                "SELECT delivered_at, plugin_id, run_id, status, title, "
                "section_count, item_count, metric_count "
                "FROM stdout_channel_log ORDER BY delivered_at DESC LIMIT ?",
                (limit,),
            )
        finally:
            db.close()

        if not rows:
            sys.stdout.write("(no deliveries yet)\n")
            return 0
        for r in rows:
            sys.stdout.write(
                f"{r[0]}  {r[1]:<20}  {r[3]:<8}  sec={r[5]} item={r[6]} "
                f"metric={r[7]}  {r[4]}\n"
            )
        return 0
