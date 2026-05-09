Output docs/opencode-llama-demo.gif

Require opencode-llama
Require opencode

Set Shell "bash"
Set FontSize 24
Set FontFamily "JetBrains Mono"
Set Width 1200
Set Height 720
Set Padding 18
Set Framerate 30
Set Theme "iTerm2 Tango Dark"
Set BorderRadius 8
Set TypingSpeed 25ms 

Hide
Type 'DEMO_PROJECT="$PWD"; DEMO_HOME="$(mktemp -d)"; mkdir "$DEMO_HOME/this-project"; cp -a "$DEMO_PROJECT/." "$DEMO_HOME/this-project"; cd "$DEMO_HOME"'
Enter
Type "export PS1='> '"
Enter
Type "clear"
Enter
Show

Type "cd this-project"
Sleep 0.5s
Enter
Sleep 1s

Type "opencode-llama"
Sleep 0.8s
Enter
Sleep 5s

Type "Say hello from the local model."
Sleep 0.8s
Enter
Sleep 8s

Ctrl+C
