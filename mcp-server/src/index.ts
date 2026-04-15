import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

const PatientDataSchema = z.object({
  patient_id: z.string(),
});

type PatientDataRequest = z.infer<typeof PatientDataSchema>;

type LabResults = {
  HbA1c: string;
  ALT: string;
  AST: string;
  eGFR: string;
};

type PatientRecord = {
  name: string;
  age: number;
  lab_results: LabResults;
  doctor_notes: string;
};

const PATIENT_DATA: Record<string, PatientRecord> = {
  P001: {
    name: "John Doe",
    age: 45,
    lab_results: {
      HbA1c: "7.2%",
      ALT: "45 U/L",
      AST: "38 U/L",
      eGFR: "85 mL/min/1.73m2",
    },
    doctor_notes:
      "Patient has history of Type 2 Diabetes. Shows interest in clinical trials for new glucose management medications. No history of cardiovascular disease. Currently stable.",
  },
  P002: {
    name: "Jane Smith",
    age: 58,
    lab_results: {
      HbA1c: "8.5%",
      ALT: "110 U/L",
      AST: "95 U/L",
      eGFR: "55 U/L",
    },
    doctor_notes:
      "Patient presents with elevated liver enzymes. Possible non-alcoholic fatty liver disease (NAFLD). Chronic kidney disease Stage 3a. Not suitable for trials requiring high renal clearance.",
  },
};

const server = new Server(
  {
    name: "patient-data-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "get_patient_data",
        description:
          "Fetch lab results and doctor notes for a specific patient ID from the legacy SQL database.",
        inputSchema: {
          type: "object",
          properties: {
            patient_id: {
              type: "string",
              description: "The unique identifier for the patient (e.g., P001, P002)",
            },
          },
          required: ["patient_id"],
        },
      },
    ],
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name !== "get_patient_data") {
    throw new Error("Tool not found");
  }

  const { patient_id }: PatientDataRequest = PatientDataSchema.parse(
    request.params.arguments,
  );
  const data = PATIENT_DATA[patient_id];

  if (!data) {
    return {
      content: [
        {
          type: "text",
          text: `No data found for patient ID: ${patient_id}`,
        },
      ],
      isError: true,
    };
  }

  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(data, null, 2),
      },
    ],
  };
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Patient Data MCP server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error in main():", error);
  process.exit(1);
});
