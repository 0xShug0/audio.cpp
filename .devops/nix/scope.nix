{
  lib,
  newScope,
  version,
  python-scripts,
}:

lib.makeScope newScope (self: {
  audiocpp = self.callPackage ./package.nix {
    inherit version python-scripts;
  };
})
