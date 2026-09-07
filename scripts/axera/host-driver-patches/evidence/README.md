# Evidence

- `fix3-silent-reset-20260907T0609.txt` — the Thunderbolt disconnect that reset
  the box, ending at `[heartbeat_recv_thread, 573]: device 3: dead!` with no
  Oops and no further output. This is fix 3's crash.
- `fix3-verified-survival-20260907T0635.txt` — the same disconnect/reconnect
  after fix 3, survived: `dev 3 offline: cached handles dropped`, uptime
  unbroken.

## Not copied here

Fix 2's kdump-captured Oops (`axcl_pcie_ioctl+0x842`, `CR2=0`, `Comm: axcl-smi`)
lives in root-only kdump output and a working copy was lost to a `/tmp` wipe on
reboot. The original is still on disk:

```sh
sudo sed -n '1590,1645p' /var/crash/202609070547/dmesg.202609070547
```

The key lines are quoted in `../NOTES.md` (Fix 2). Note that `/var/crash` also
holds `dump-incomplete` files that never finished writing — the 06:09 silent
reset produced **no** vmcore at all, which was itself diagnostic.
