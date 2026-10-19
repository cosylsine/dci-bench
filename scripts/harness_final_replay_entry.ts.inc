// Appended to a hash-verified copy of dci-run.ts, never to the live runner.
// Reuse its message conversion, budget estimates, response parsing and ID gates.
type ReplayInput = {
	schema_version: string;
	task: string;
	query_id: string;
	origin_profile: string;
	messages: Message[];
	read_receipts: EvidenceState["readReceipts"];
	corpus_filenames: string[];
	control: { turn: number; remaining_tokens: number; remaining_seconds: number; reason: string };
	original_final_prompt: string;
	prefix_sha256: string;
	wire_prefix: Record<string, unknown>[];
	wire_prefix_sha256: string;
	request_template: Record<string, unknown>;
	original_request_sha256: string;
};

function replaySetup(input: ReplayInput) {
	if (input.schema_version !== "dci-final-replay-input-v1" || !DEFAULT_CONTRACT.budget_control ||
		!["budgeted-stable", "budgeted-selective"].includes(PROTOCOL_PROFILE)) throw new Error("Invalid replay contract");
	const assistants = input.messages.filter((m): m is AssistantMessage => m.role === "assistant");
	const usages = assistants.map(readMessageUsage);
	const used = usages.reduce((sum, u) => sum + u.value.input + u.value.output, 0);
	const prefixHash = createHash("sha256").update(JSON.stringify(sortJsonKeys(input.messages))).digest("hex");
	const wireHash = createHash("sha256").update(JSON.stringify(sortJsonKeys(input.wire_prefix))).digest("hex");
	if (prefixHash !== input.prefix_sha256 || usages.some(u => !u.available) ||
		wireHash !== input.wire_prefix_sha256 ||
		assistants.length !== input.control.turn || used !== DEFAULT_CONTRACT.max_total_model_tokens - input.control.remaining_tokens ||
		input.control.remaining_seconds <= 0 || input.control.remaining_seconds > DEFAULT_CONTRACT.sample_timeout_seconds ||
		input.read_receipts.length === 0) throw new Error("Replay prefix, usage or evidence mismatch");
	const policy = DEFAULT_CONTRACT.budget_control;
	const prompt = policy.final_prompt.replace("{reason}", input.control.reason)
		.replace("{remaining_tokens}", String(input.control.remaining_tokens));
	if (input.origin_profile === PROTOCOL_PROFILE && prompt !== input.original_final_prompt) {
		throw new Error("Original final prompt cannot be reproduced");
	}
	const original = { ...input.request_template, messages: [...input.wire_prefix, { role: "user", content: input.original_final_prompt }] };
	if (createHash("sha256").update(JSON.stringify(sortJsonKeys(original))).digest("hex") !== input.original_request_sha256 ||
		input.request_template.tools !== null || input.request_template.tool_choice !== null) {
		throw new Error("Saved wire request mismatch");
	}
	const context: Context = { systemPrompt: DEFAULT_CONTRACT.system_prompt, messages: structuredClone(input.messages), tools: [] };
	const evidence: EvidenceState = { bodyReadFiles: new Set(input.read_receipts.map(r => r.path)),
		contentSearchCommands: [], readReceipts: input.read_receipts };
	return { used, prompt, context, evidence, corpusIds: corpusDocumentIds(input.corpus_filenames) };
}

async function replayMain() {
	const inputPath = process.argv[2];
	if (inputPath === "--check-inputs") {
		const paths = JSON.parse(readFileSync(process.argv[3], "utf8")) as string[];
		for (const path of paths) {
			const input = JSON.parse(readFileSync(path, "utf8")) as ReplayInput;
			let setup: ReturnType<typeof replaySetup>;
			try { setup = replaySetup(input); } catch (error) { throw new Error(`${path}: ${String(error)}`); }
			const estimated = estimateNextInputTokens(setup.context, true) + Buffer.byteLength(setup.prompt, "utf8") +
				DEFAULT_CONTRACT.max_model_output_tokens + CONTEXT_SAFETY_TOKENS;
			if (estimated >= input.control.remaining_tokens) throw new Error(`Replay final cannot fit: ${path}`);
			setup.context.messages.push({ role: "user", content: [{ type: "text", text: setup.prompt }], timestamp: 0 });
			if (estimateNextInputTokens(setup.context) + DEFAULT_CONTRACT.max_model_output_tokens + CONTEXT_SAFETY_TOKENS > 65536) {
				throw new Error(`Replay context cannot fit: ${path}`);
			}
		}
		console.log(JSON.stringify({ profile: PROTOCOL_PROFILE, contract_sha256: CONTRACT_SHA256, inputs: paths.length, status: "ok" }));
		return;
	}
	const input = JSON.parse(readFileSync(inputPath, "utf8")) as ReplayInput;
	const output = process.argv[3];
	const setup = replaySetup(input);
	const { context, evidence, corpusIds } = setup;
	let totalModelTokens = setup.used;
	let turns = input.control.turn;
	let prompt = setup.prompt;
	let failure: StructuredFailure = null;
	const repairReasons: string[] = [];
	const requests: Record<string, unknown>[] = [];
	const started = Date.now();
	const deadline = started + input.control.remaining_seconds * 1000;
	const abort = new AbortController();
	const timer = setTimeout(() => abort.abort(), Math.max(1, deadline - Date.now()));
	const args = parseArgs(["--query-file", inputPath, "--workspace", ".", "--output", output,
		"--context-window", "65536", "--model", process.env.SERVED_MODEL_NAME ?? "Qwen3.8-27B",
		"--openai-api-key", "replay",
		"--openai-base-url", process.env.BACKEND_BASE_URL ?? "http://127.0.0.1:28099/v1"]);
	const model = createModel("bridge", args.model, args.openaiBaseUrl, 65536);
	const streamFn = createBridgeStreamFn(args);
	// Capture the actual requests before transport. No tools are ever instantiated.
	const transport = globalThis.fetch;
	globalThis.fetch = async (url, options) => {
		const bridge = JSON.parse(String(options?.body)) as { tools: unknown; tool_choice: unknown; messages: Record<string, unknown>[] };
		if (bridge.tool_choice !== "none" || JSON.stringify(bridge.tools) !== "[]") throw new Error("Replay enabled tools");
		const generated = bridge.messages.slice(input.messages.length + 1).map(message =>
			message.role === "assistant" ? { role: "assistant", content: message.content ?? "" } : message);
		const request = { ...input.request_template, messages: [...input.wire_prefix, ...generated] };
		requests.push(request);
		return transport(url, { ...options, body: JSON.stringify(request) });
	};
	let finalText = "";
	let parsed: ParsedFinal | undefined;
	try {
		while (!failure) {
			if (Date.now() >= deadline) { failure = makeFailure("timeout", "replay remaining wall deadline reached"); break; }
			const estimated = estimateNextInputTokens(context, true) + Buffer.byteLength(prompt, "utf8") +
				DEFAULT_CONTRACT.max_model_output_tokens + CONTEXT_SAFETY_TOKENS;
			if (estimated >= DEFAULT_CONTRACT.max_total_model_tokens - totalModelTokens) {
				failure = makeFailure("token_limit", "remaining cumulative budget cannot safely fund finalization or repair"); break;
			}
			context.messages.push({ role: "user", content: [{ type: "text", text: prompt }], timestamp: Date.now() });
			const stream = await streamFn(model, context, { toolChoice: "none", signal: abort.signal });
			const message = await stream.result();
			context.messages.push(message);
			turns++;
			const current = readMessageUsage(message);
			if (!current.available) {
				failure = Date.now() >= deadline ? makeFailure("timeout", "replay remaining wall deadline reached") :
					message.stopReason === "error" ? failureFromAgentError(message.errorMessage ?? "agent returned an error") :
					makeFailure("usage_unavailable", "model usage was missing or incomplete"); break;
			}
			totalModelTokens += current.value.input + current.value.output;
			if (Date.now() >= deadline) { failure = makeFailure("timeout", "sample wall deadline reached before final acceptance"); break; }
			if (current.value.output > DEFAULT_CONTRACT.max_model_output_tokens) {
				failure = makeFailure("agent_error", "backend exceeded per-turn output token limit"); break;
			}
			if (totalModelTokens >= DEFAULT_CONTRACT.max_total_model_tokens) {
				failure = makeFailure("token_limit", "cumulative model input/output tokens reached 128000"); break;
			}
			finalText = finalAnswerFrom(context.messages);
			parsed = parseFinalJson(finalText);
			// Preserve the original stop-check ordering: parsed finals reach ID validation
			// even on the last permitted turn; malformed finals hit the step guard.
			if (!parsed && turns >= DEFAULT_CONTRACT.max_agent_steps) {
				failure = makeFailure("max_agent_steps", "max_agent_steps reached before an acceptable final answer"); break;
			}
			const issue = finalAcceptanceIssue(parsed, evidence, corpusIds);
			if (!issue) break;
			if (turns >= DEFAULT_CONTRACT.max_agent_steps || repairReasons.length >= DEFAULT_CONTRACT.max_final_repair_turns) {
				failure = makeFailure("invalid_json", issue); break;
			}
			repairReasons.push(issue);
			prompt = buildRepairPrompt(issue, true);
		}
	} finally {
		clearTimeout(timer);
		globalThis.fetch = transport;
	}
	const valid = !failure && parsed !== undefined;
	finalText = finalAnswerFrom(context.messages);
	writeJsonAtomicSync(output, {
		schema_version: "dci-final-replay-result-v1", task: input.task, query_id: input.query_id,
		origin_profile: input.origin_profile, profile: PROTOCOL_PROFILE, base_contract_sha256: CONTRACT_SHA256,
		prefix_sha256: input.prefix_sha256, wire_prefix_sha256: input.wire_prefix_sha256, valid_output: valid, failure,
		ranked_doc_ids: valid ? parsed!.ranked_doc_ids : [], raw_final_text: finalText,
		final_output_adapter: valid ? parsed!.adapter ?? null : null,
		repair_attempts: repairReasons.length, repair_reasons: repairReasons,
		prefix_usage: usageSummary(input.messages, setup.used),
		usage: usageSummary(context.messages.slice(input.messages.length), totalModelTokens - setup.used),
		cumulative_model_tokens: totalModelTokens, prefix_turns: input.control.turn, agent_steps: turns,
		wall_time_seconds: (Date.now() - started) / 1000, allowed_wall_seconds: input.control.remaining_seconds,
		requests, generated_messages: context.messages.slice(input.messages.length), tool_executions: 0,
	});
}
