use std::net::SocketAddr;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) != Some("serve") {
        eprintln!("usage: lgbox serve --port P --workers N --queue Q");
        std::process::exit(2);
    }
    let mut port: u16 = 7777;
    let mut workers: usize = 4;
    let mut queue: usize = 64;
    let mut i = 1;
    while i < args.len() {
        let v = match args.get(i + 1) {
            Some(v) => v.as_str(),
            None => die(&format!("missing value for {}", args[i])),
        };
        match args[i].as_str() {
            "--port" => port = v.parse().unwrap_or_else(|e: std::num::ParseIntError| die(&e.to_string())),
            "--workers" => workers = v.parse().unwrap_or_else(|e: std::num::ParseIntError| die(&e.to_string())),
            "--queue" => queue = v.parse().unwrap_or_else(|e: std::num::ParseIntError| die(&e.to_string())),
            other => die(&format!("unknown option {other}")),
        }
        i += 2;
    }
    if workers == 0 || queue == 0 {
        die("workers and queue must be >= 1");
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async move {
        let addr: SocketAddr = format!("0.0.0.0:{port}").parse().unwrap();
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
        println!(
            "lgbox listening on {} workers={workers} queue={queue}",
            listener.local_addr().unwrap()
        );
        axum::serve(listener, lgbox::app(workers, queue)).await.unwrap();
    });
}

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2);
}
