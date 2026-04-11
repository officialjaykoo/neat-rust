use std::io::{self, BufRead, Write};

use json::JsonValue;
use neat_rust::compat::js::{
    evaluate_policy_batch, CompiledPolicyRequest, CompiledPolicySpec, PolicyBridgeBackend,
};

enum BridgeRequest {
    Init {
        id: u64,
        compiled: CompiledPolicySpec,
    },
    Eval {
        id: u64,
        request: CompiledPolicyRequest,
    },
    Shutdown {
        id: u64,
    },
}

fn main() {
    if let Err(message) = run() {
        eprintln!("{message}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let backend = parse_backend(std::env::args().skip(1).collect())?;
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    let mut compiled: Option<CompiledPolicySpec> = None;

    for line in stdin.lock().lines() {
        let line = line.map_err(|err| err.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let request = parse_request(&line)?;
        match request {
            BridgeRequest::Init { id, compiled: next } => {
                compiled = Some(next);
                write_response(&mut stdout, id, Ok(JsonValue::new_object()))?;
            }
            BridgeRequest::Eval { id, request } => {
                let Some(spec) = compiled.as_ref() else {
                    write_response(
                        &mut stdout,
                        id,
                        Err("bridge received eval before init".to_string()),
                    )?;
                    continue;
                };
                match evaluate_policy_batch(spec, &request, backend) {
                    Ok(result) => write_response(&mut stdout, id, Ok(result.to_json_value()))?,
                    Err(err) => write_response(&mut stdout, id, Err(err.to_string()))?,
                }
            }
            BridgeRequest::Shutdown { id } => {
                write_response(&mut stdout, id, Ok(JsonValue::new_object()))?;
                break;
            }
        }
    }

    Ok(())
}

fn parse_request(text: &str) -> Result<BridgeRequest, String> {
    let value = json::parse(text).map_err(|err| format!("invalid request JSON: {err}"))?;
    let kind = value["kind"]
        .as_str()
        .ok_or_else(|| "request.kind must be a string".to_string())?;
    let id = value["id"]
        .as_u64()
        .ok_or_else(|| "request.id must be a u64".to_string())?;
    match kind {
        "init" => Ok(BridgeRequest::Init {
            id,
            compiled: CompiledPolicySpec::from_json(&value["compiled"])
                .map_err(|err| err.to_string())?,
        }),
        "eval" => Ok(BridgeRequest::Eval {
            id,
            request: CompiledPolicyRequest::from_json(&value).map_err(|err| err.to_string())?,
        }),
        "shutdown" => Ok(BridgeRequest::Shutdown { id }),
        other => Err(format!("unsupported bridge request kind {other:?}")),
    }
}

fn parse_backend(args: Vec<String>) -> Result<PolicyBridgeBackend, String> {
    let mut idx = 0usize;
    while idx < args.len() {
        if args[idx] == "--backend" {
            let value = args.get(idx + 1).cloned().unwrap_or_default();
            return PolicyBridgeBackend::parse(&value).map_err(|err| err.to_string());
        }
        idx += 1;
    }
    Ok(PolicyBridgeBackend::Auto)
}

fn write_response(
    stdout: &mut impl Write,
    id: u64,
    payload: Result<JsonValue, String>,
) -> Result<(), String> {
    let mut response = JsonValue::new_object();
    response["id"] = id.into();
    match payload {
        Ok(value) => {
            response["ok"] = true.into();
            if value.is_object() {
                for (key, item) in value.entries() {
                    response[key] = item.clone();
                }
            }
        }
        Err(error) => {
            response["ok"] = false.into();
            response["error"] = error.into();
        }
    }
    let line = response.dump();
    stdout
        .write_all(line.as_bytes())
        .and_then(|_| stdout.write_all(b"\n"))
        .and_then(|_| stdout.flush())
        .map_err(|err| err.to_string())
}
