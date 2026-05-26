Known issues
===========

Linux: glibc requirement
------------------------
On Linux, the runtime requires GNU C Library (glibc) version 2.39 or newer.
This is typically needed by native dependencies used by Kratos and related tooling.

How to check your glibc version
...............................
Run one of the following in a terminal (they may differ per distro):

- ldd --version

You should see a version like "glibc 2.39" or newer.

Upgrading glibc (high-level guidance)
.....................................
It is recommended to upgrade your distribution to a release that ships glibc >= 2.39.
Most recent Linux distributions include this version or newer by default.

Notes
.....
- If you can't upgrade the host OS, consider using a Pod or Docker container that matches the required
  glibc version and provides Kratos and dependencies. You can find a Docker file to run STEM
  in the `STEM repository <https://github.com/StemVibrations/STEM/blob/main/podman/>`__.
- After upgrading, re-create your Python virtual environment to avoid binary incompatibilities.