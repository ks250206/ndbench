import * as mlMatrix from "./ml_matrix_backend";
import * as raw from "./raw_backend";
import {
  defaultIterations,
  defaultSize,
  OPERATIONS,
  Operation,
} from "./shared";

type Backend = "raw" | "ml-matrix";

interface Config {
  backend: Backend;
  operation: Operation;
  size: number;
  iterations: number;
}

function usage(): string {
  return `Usage:
  node dist/ndbench.js [--backend <raw|ml-matrix>] --op <operation> [options]

Operations:
  vector2, vector3    low-dimensional vector arithmetic repeated --iterations times
  affine2, affine3    homogeneous affine transform of --size points
  matvec              dense matrix-vector product of a --size square matrix
  matmul              dense square matrix product of order --size
  cosine1024          cosine similarity of two f64 vectors of length 1024
  eigh                full symmetric eigendecomposition of order --size

Options:
  --backend <NAME>    raw (default) or ml-matrix
  --size <N>          point count or matrix order, depending on operation
  --iterations <N>    repeat the operation N times
  --help              show this message

Examples:
  node dist/ndbench.js --backend raw --op vector3 --iterations 1000000
  node dist/ndbench.js --backend ml-matrix --op matmul --size 256
  node dist/ndbench.js --backend ml-matrix --op eigh --size 128
`;
}

function parseOperation(value: string): Operation {
  const aliases: Record<string, Operation> = {
    vec2: "vector2",
    vec3: "vector3",
    cosine: "cosine1024",
    diagonalize: "eigh",
  };
  const operation = aliases[value] ?? value;
  if ((OPERATIONS as readonly string[]).includes(operation)) {
    return operation as Operation;
  }
  throw new Error(
    `unknown operation \`${value}\`; expected ${OPERATIONS.join(", ")}`,
  );
}

function parseBackend(value: string): Backend {
  if (value === "raw" || value === "native") {
    return "raw";
  }
  if (value === "ml-matrix" || value === "mlmatrix" || value === "ml_matrix") {
    return "ml-matrix";
  }
  throw new Error("unknown backend `" + value + "`; expected raw or ml-matrix");
}

function parsePositiveInteger(flag: string, value: string | undefined): number {
  if (value === undefined) {
    throw new Error(`missing value after ${flag}`);
  }
  if (!/^\d+$/.test(value)) {
    throw new Error(`invalid value for ${flag}: ${value}`);
  }
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`${flag} must be a positive safe integer`);
  }
  return parsed;
}

function parseArgs(argv: readonly string[]): Config | undefined {
  let backend: Backend = "raw";
  let operation: Operation | undefined;
  let size: number | undefined;
  let iterations: number | undefined;

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    switch (argument) {
      case "--help":
      case "-h":
        return undefined;
      case "--backend":
        backend = parseBackend(argv[++index]);
        break;
      case "--op":
      case "--operation":
        operation = parseOperation(argv[++index]);
        break;
      case "--size":
        size = parsePositiveInteger("--size", argv[++index]);
        break;
      case "--iterations":
        iterations = parsePositiveInteger("--iterations", argv[++index]);
        break;
      default:
        throw new Error(`unknown argument \`${argument}\`\n\n${usage()}`);
    }
  }

  if (operation === undefined) {
    throw new Error(`missing --op\n\n${usage()}`);
  }

  return {
    backend,
    operation,
    size: size ?? defaultSize(operation),
    iterations: iterations ?? defaultIterations(operation),
  };
}

function run(config: Config): number {
  const implementation = config.backend === "raw" ? raw : mlMatrix;
  switch (config.operation) {
    case "vector2":
      return implementation.vector2(config.iterations);
    case "vector3":
      return implementation.vector3(config.iterations);
    case "affine2":
      return implementation.affine2(config.size, config.iterations);
    case "affine3":
      return implementation.affine3(config.size, config.iterations);
    case "matvec":
      return implementation.matvec(config.size, config.iterations);
    case "matmul":
      return implementation.matmul(config.size, config.iterations);
    case "cosine1024":
      return implementation.cosine1024(config.iterations);
    case "eigh":
      return implementation.eigh(config.size, config.iterations);
  }
}

function formatChecksum(value: number): string {
  const exponential = value.toExponential(17);
  return exponential.replace(/e([+-])(\d+)$/, (_match, sign: string, exponent: string) =>
    `e${sign}${exponent.padStart(2, "0")}`,
  );
}

function main(): number {
  try {
    const config = parseArgs(process.argv.slice(2));
    if (config === undefined) {
      console.log(usage());
      return 0;
    }
    console.log(`checksum=${formatChecksum(run(config))}`);
    return 0;
  } catch (error) {
    console.error(`error: ${error instanceof Error ? error.message : String(error)}`);
    return 1;
  }
}

process.exitCode = main();
