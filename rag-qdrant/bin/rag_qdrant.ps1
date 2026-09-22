# PowerShell runner for rag_qdrant CLI
$ScriptDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$env:PYTHONPATH = "$ScriptDir;$env:PYTHONPATH"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

try {
    python -m rag_qdrant.cli @args
} catch {
    # Handled inside Python via KeyboardInterrupt
}
