use std::io::{self, BufRead, Write};

use neat_rust::runtime::{
    evaluate_policy_batch, CompiledPolicyRequest, CompiledPolicySpec, PolicyBridgeBackend,
};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum BridgeRequest {
    Init {
        id: u64,
        compiled: CompiledPolicySpec,
    },
    Eval {
        id: u64,
        #[serde(flatten)]
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
                next.validate().map_err(|err| err.to_string())?;
                compiled = Some(next);
                write_response(&mut stdout, id, Ok(json!({})))?;
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
                    Ok(result) => {
                        let value = serde_json::to_value(result).map_err(|err| err.to_string())?;
                        write_response(&mut stdout, id, Ok(value))?;
                    }
                    Err(err) => write_response(&mut stdout, id, Err(err.to_string()))?,
                }
            }
            BridgeRequest::Shutdown { id } => {
                write_response(&mut stdout, id, Ok(json!({})))?;
                break;
            }
        }
    }

    Ok(())
}

fn parse_request(text: &str) -> Result<BridgeRequest, String> {
    serde_json::from_str(text).map_err(|err| format!("invalid request JSON: {err}"))
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
    payload: Result<Value, String>,
) -> Result<(), String> {
    let mut response = json!({ "id": id });
    let response_object = response
        .as_object_mut()
        .ok_or_else(|| "internal response JSON was not an object".to_string())?;
    match payload {
        Ok(value) => {
            response_object.insert("ok".to_string(), Value::Bool(true));
            if let Some(payload_object) = value.as_object() {
                for (key, item) in payload_object {
                    response_object.insert(key.clone(), item.clone());
                }
            }
        }
        Err(error) => {
            response_object.insert("ok".to_string(), Value::Bool(false));
            response_object.insert("error".to_string(), Value::String(error));
        }
    }
    let line = serde_json::to_string(&response).map_err(|err| err.to_string())?;
    stdout
        .write_all(line.as_bytes())
        .and_then(|_| stdout.write_all(b"\n"))
        .and_then(|_| stdout.flush())
        .map_err(|err| err.to_string())
}
