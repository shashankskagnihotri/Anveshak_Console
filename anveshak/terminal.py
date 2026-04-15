"""Terminal user interface for SSH-first Anveshak sessions."""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console

from .chat.service import ChatService


class TerminalChat:
    """Simple REPL for chatting with Anveshak from a shell."""

    def __init__(self, service: ChatService) -> None:
        self.service = service
        self.console = Console()
        self.session = service.create_session()
        self.pending_paths: list[Path] = []

    def run(self) -> None:
        """Start the blocking terminal loop and stream responses inline."""

        self.console.print("[bold]Anveshak Console[/bold]")
        self._wait_for_runtime()
        self.console.print("[dim]Loading the reasoning model into GPU/CPU memory before the first prompt[/dim]")
        self.service.wait_until_model_ready()
        self.console.print("[dim]Model warm and ready.[/dim]")
        self.console.print("Commands: /attach <paths...>, /files, /clear, /obliviate, /exit")
        while True:
            try:
                prompt = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\nExiting.")
                return

            if not prompt:
                continue
            if prompt == "/exit":
                return
            if prompt.startswith("/attach "):
                self.pending_paths.extend(Path(item).expanduser().resolve() for item in prompt.split()[1:])
                self.console.print(f"Queued {len(self.pending_paths)} attachment(s).")
                continue
            if prompt == "/files":
                if not self.pending_paths:
                    self.console.print("No pending attachments.")
                else:
                    for path in self.pending_paths:
                        self.console.print(str(path))
                continue
            if prompt == "/clear":
                self.pending_paths.clear()
                self.console.print("Cleared pending attachments.")
                continue
            if prompt == "/obliviate":
                prompt = "Obliviate"

            attachments = self.service.save_uploads(self.session.session_id, self.pending_paths)
            handle = self.service.submit_message(
                session_id=self.session.session_id,
                text=prompt,
                attachments=attachments,
            )
            self.pending_paths.clear()
            self._render_run(handle.run_id)

    def _render_run(self, run_id: str) -> None:
        """Render one streamed run in the terminal as events arrive."""

        handle = self.service.runs[run_id]
        answer_started = False
        while True:
            event = handle.next_event(timeout=0.5)
            if event is None:
                if handle.done:
                    break
                continue

            payload = event.payload
            if event.event_type == "status":
                self.console.print(f"[dim]{payload.get('text', '')}[/dim]")
            elif event.event_type == "reasoning":
                self.console.print(payload.get("text", ""), style="cyan", end="")
            elif event.event_type == "token":
                if not answer_started:
                    self.console.print("\nAssistant> ", style="bold green", end="")
                    answer_started = True
                self.console.print(payload.get("text", ""), style="green", end="")
            elif event.event_type == "warning":
                self.console.print(f"[yellow]{payload.get('text', '')}[/yellow]")
            elif event.event_type == "transcription":
                attachment_name = payload.get("attachment_name", "audio")
                self.console.print(f"[bold magenta]Transcript ({attachment_name})>[/bold magenta] {payload.get('text', '')}")
            elif event.event_type == "done":
                self.console.print()
                citations = payload.get("citations", [])
                if citations:
                    self.console.print("[bold]Sources[/bold]")
                    for citation in citations[:12]:
                        label = citation.get("label", citation.get("source_id", "source"))
                        metadata = citation.get("metadata", {})
                        target = metadata.get("source_path") or metadata.get("url") or ""
                        self.console.print(f"- {label}: {target}")
                break
            elif event.event_type == "error":
                self.console.print(f"[red]Error:[/red] {payload.get('message', 'unknown error')}")
                break

    def _wait_for_runtime(self) -> None:
        """Poll runtime preparation until the backend is ready for prompts."""

        last_message = None
        while True:
            status = self.service.runtime_status()
            message = status.get("message", "")
            if message and message != last_message:
                percent = int(status.get("progress", 0.0) * 100)
                self.console.print(f"[dim]{message} ({percent}%)[/dim]")
                last_message = message
            if status.get("ready"):
                return
            if status.get("phase") == "error":
                raise RuntimeError(status.get("error") or "Runtime preparation failed")
            time.sleep(0.5)
