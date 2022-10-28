import UIKit

class ViewController: UIViewController {

    @IBOutlet weak var textBox: UITextView!
    @IBOutlet weak var predictedReviewLabel: UILabel!
    
    // url set to local
    var urlComponents = URLComponents(string: "http://127.0.0.1:8000")
    
    // HTTP response object
    struct ResponseData: Decodable {
        let prediction: Double
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        predictedReviewLabel.text = ""
        urlComponents!.queryItems = [URLQueryItem(name: "q", value: "")]
    }

    // generates Positive or Negative for the user's movie review
    @IBAction func buttonPressed(_ sender: Any) {
        // do not send HTTP request if no input
        if textBox.text.trimmingCharacters(in: .whitespacesAndNewlines) == "" {
            return
        }

        // send HTTP GET request and pass input text as query
        urlComponents!.queryItems = [URLQueryItem(name: "q", value: textBox.text)]
        let task = URLSession.shared.dataTask(with: urlComponents!.url!, completionHandler: {
            data, _, error in
            if let data = data {
                if let responseData = try? JSONDecoder().decode(ResponseData.self, from: data) {
                    // update label with response
                    DispatchQueue.main.async() {
                        self.predictedReviewLabel.text = responseData.prediction == 1 ? "Positive" : "Negative"
                    }
                } else {
                    print("Invalid Response")
                }
            } else if let error = error {
                print("HTTP Request Failed \(error.localizedDescription)")
            }
        })
        task.resume()
    }
}
