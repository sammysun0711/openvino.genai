from http.server import HTTPServer, BaseHTTPRequestHandler

HOST = "localhost"
PORT = 8080

class MyHandler(BaseHTTPRequestHandler):
  def do_POST(self):
    # Get the request body data
    content_length = int(self.headers.get('Content-Length', 0))
    data = self.rfile.read(content_length).decode("utf-8")

    # Process the received data
    print(f"Received data from client: {data}")

    # Send a response (optional)
    self.send_response(200)
    self.send_header('Content-type', 'text/plain')
    self.end_headers()
    self.wfile.write(bytes("Data received!", "utf-8"))

with HTTPServer((HOST, PORT), MyHandler) as server:
  print(f"Server listening on http://{HOST}:{PORT}")
  server.serve_forever()