This is the source of my melonDS switch version.

It's a mess and not finished. Over the next time, I want to try to integrate the ARM64 JIT and the ARM Neon code(of course not as it's the current state!) into melonDS, maybe the Android port benefits from this too.

Credits:
- Arisotura, obviously for melonDS
- Dolphin people, of whom I've taken the JIT code emitter and who helped me on IRC!
- Hydr8gon, who did the original melonDS switch port (from which this ports borrows quite a lot of code)
- [dear imgui](https://github.com/ocornut/imgui)
- endrift, the cmake switch buildscript is based of the of mgba
- libnx and devkitpro people