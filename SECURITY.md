# Security policy

## Supported versions

Security fixes are provided for the latest released minor version. Pre-release
builds and unpublished native artifacts are not supported deployments.

## Reporting a vulnerability

Use GitHub's private security-advisory feature for this repository. Do not open
a public issue for suspected code execution, artifact substitution, unsafe
checkpoint loading, dependency compromise, credential disclosure, or denial of
service.

Include the affected version/commit, platform, reproduction, impact, and any
known mitigation. Maintainers will acknowledge a complete report within seven
days and coordinate disclosure after a fix is available.

## Artifact trust

Install portable packages from PyPI and native wheels or containers only from
the matching GitHub release. GPU bundle installers verify recorded SHA-256
hashes and reject unknown manifest formats. PyTorch checkpoints can contain
pickle data; load only checkpoints from trusted sources.
