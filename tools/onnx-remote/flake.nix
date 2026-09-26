{
  description = "onnxsim native remote transport with ROS2 development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nix-ros-overlay = {
      url = "github:lopsided98/nix-ros-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, nix-ros-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ nix-ros-overlay.overlays.default ];
        };
        ros = pkgs.rosPackages.humble;
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cmake
            pkg-config
            ros.ament-cmake
            ros.rclcpp
            ros.std-msgs
            ros.std-srvs
            ros.ros2cli
          ];
          shellHook = ''
            export ROS_DISTRO=humble
            export ROS_VERSION=2
            echo "onnx-remote ROS2 shell: build with -DONNXSIM_REMOTE_ROS2=ON"
          '';
        };
      });
}
