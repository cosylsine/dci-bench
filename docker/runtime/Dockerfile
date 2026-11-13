# Optional CI build. Evaluation nodes use the exported offline bundle, not Docker.
FROM debian@sha256:7378260c0a098d27ac206225db5e7d92b8c7c84ea7afccc86eeb513756bbd392
RUN printf '%s\n' \
      'Types: deb' \
      'URIs: http://snapshot.debian.org/archive/debian/20261104T212339Z' \
      'Suites: trixie' 'Components: main' \
      'Signed-By: /usr/share/keyrings/debian-archive-keyring.pgp' \
      'Check-Valid-Until: no' > /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends ripgrep=14.1.1-1+b4 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /var/log/* \
    && mkdir -p /workspace/corpus /scratch /tmp /sandbox-bin \
    && touch /sandbox-bin/rg.conf
