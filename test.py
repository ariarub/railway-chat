import unittest
from app import app
import timeit
import chatbot

class TestChatbot(unittest.TestCase):

    # Test the home page
    def test_home(self):
        tester = app.test_client(self)
        response = tester.get("/")
        status_code = response.status_code
        self.assertEqual(status_code, 200)

    # Test a valid user message
    def test_valid_message(self):
        tester = app.test_client(self)
        sentence = "hi"
        response = tester.get("/get?msg={}".format(sentence))
        response_text = response.get_data(as_text=True)
        possibleresponses = ["Hello", "Hi", "What can I do for you?"]
        self.assertIn(response_text, possibleresponses)

    # Test an invalid user message
    def test_invalid_message(self):
        tester = app.test_client(self)
        response = tester.get("/get?msg=invalid")
        response_text = response.get_data(as_text=True)
        self.assertIn("I'm sorry I don't understand what you are asking. Please can you rephrase your question or consider this website for further help https://www.phonak.com/au/en/support/product-support.html.", response_text)

    # Test an empty user message
    def test_empty_message(self):
        tester = app.test_client(self)
        response = tester.get("/get?msg=")
        response_text = response.get_data(as_text=True)
        self.assertIn("I'm sorry I don't understand what you are asking. Please can you rephrase your question or consider this website for further help https://www.phonak.com/au/en/support/product-support.html.", response_text)

    def test_response_time(self):
        sentence = "Hello"
        start_time = timeit.default_timer()
        intents = chatbot.predict_class(sentence)
        response = chatbot.get_response(intents, chatbot.intents)
        elapsed = timeit.default_timer() - start_time
        print("Response time:", elapsed)
        self.assertLess(elapsed, 0.5)


if __name__ == "__main__":
    unittest.main()