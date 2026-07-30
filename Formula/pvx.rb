class Pvx < Formula
  include Language::Python::Virtualenv

  desc "Phase-vocoder DSP toolkit with pvx command suite"
  homepage "https://github.com/TheColby/pvx"
  url "https://github.com/TheColby/pvx/archive/refs/tags/v0.1.0a1.tar.gz"
  version "0.1.0a1"
  sha256 "209fb21872fab1571727657b24a3c5d21660ae46adb2a3255b827ae35495746f"
  license "MIT"

  head "https://github.com/TheColby/pvx.git", branch: "main"

  depends_on "libsndfile"
  depends_on "python@3.12"
  uses_from_macos "libffi"

  def install
    venv = virtualenv_create(libexec, "python3.12")

    # The project tap owns this formula. Runtime packages are isolated in
    # libexec so they cannot conflict with a user's Python environment.
    venv.pip_install %w[
      numpy>=1.24
      soundfile>=0.12.1
      scipy>=1.10
      librosa>=0.10.2
      pyloudnorm>=0.1.1
    ]
    venv.pip_install_and_link buildpath

    man1.install Dir[buildpath/"man/man1/*.1"] if (buildpath/"man/man1").exist?
  end

  test do
    %w[
      pvx
      pvxvoc
      pvxfreeze
      pvxwarp
      pvxformant
      pvxfilter
      pvxretune
      pvxanalysis
    ].each do |command|
      assert_path_exists bin/command
    end
    assert_match "Unified entry point", shell_output("#{bin}/pvx --help")
    assert_match "phase vocoder", shell_output("#{bin}/pvxvoc --help")
  end
end
