// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "ndbench-swift",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(
            name: "ndbench",
            targets: ["ndbench"]
        ),
    ],
    targets: [
        .executableTarget(
            name: "ndbench",
            path: "Sources/ndbench",
            swiftSettings: [
                // The current Accelerate C headers select the modern ILP64
                // BLAS/LAPACK declarations through C compiler definitions.
                .unsafeFlags([
                    "-Xcc", "-DACCELERATE_NEW_LAPACK",
                    "-Xcc", "-DACCELERATE_LAPACK_ILP64",
                ]),
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
            ]
        ),
    ]
)
