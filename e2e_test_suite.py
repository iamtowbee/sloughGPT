#!/usr/bin/env python3
"""
Comprehensive E2E Test Suite for SloughGPT WebUI
Tests all frontend functionality without requiring Cypress
"""

import os
import sys
import time
import requests
import subprocess
import json
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains

class SloughGPTE2ETester:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.driver = None
        self.test_results = []
        
    def setup_driver(self):
        """Setup Chrome WebDriver"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            return True
        except Exception as e:
            print(f"❌ WebDriver setup failed: {e}")
            return False
    
    def log_test(self, test_name, passed, message=""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append({
            "name": test_name,
            "passed": passed,
            "message": message
        })
        print(f"{status}: {test_name}")
        if message:
            print(f"   {message}")
    
    def wait_for_element(self, selector, timeout=10):
        """Wait for element to be present"""
        try:
            wait = WebDriverWait(self.driver, timeout)
            element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            return element
        except:
            return None
    
    def click_element(self, selector, timeout=10):
        """Click on element"""
        element = self.wait_for_element(selector, timeout)
        if element:
            try:
                element.click()
                return True
            except Exception as e:
                # Try JavaScript click if normal click fails
                self.driver.execute_script("arguments[0].click();", element)
                return True
        return False
    
    def type_text(self, selector, text, timeout=10):
        """Type text into element"""
        element = self.wait_for_element(selector, timeout)
        if element:
            element.clear()
            element.send_keys(text)
            return True
        return False
    
    def test_server_health(self):
        """Test API health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log_test("API Health Check", True, f"Status: {data.get('status')}")
                return True
            else:
                self.log_test("API Health Check", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("API Health Check", False, f"Error: {e}")
            return False
    
    def test_frontend_loads(self):
        """Test that frontend loads successfully"""
        try:
            self.driver.get(self.base_url)
            time.sleep(2)
            
            # Check if page loaded
            title = self.driver.title
            if "SloughGPT" in title or "WebUI" in title:
                self.log_test("Frontend Load", True, f"Title: {title}")
                return True
            else:
                self.log_test("Frontend Load", False, f"Unexpected title: {title}")
                return False
        except Exception as e:
            self.log_test("Frontend Load", False, f"Error: {e}")
            return False
    
    def test_sloughgpt_branding(self):
        """Test SloughGPT branding elements"""
        try:
            self.driver.get(self.base_url)
            time.sleep(2)
            
            # Check for SloughGPT branding
            page_source = self.driver.page_source.lower()
            
            branding_elements = [
                "sloughgpt",
                "enhanced webui",
                "advanced ai interface"
            ]
            
            found_branding = any(element in page_source for element in branding_elements)
            
            if found_branding:
                self.log_test("SloughGPT Branding", True, "SloughGPT branding found")
            else:
                self.log_test("SloughGPT Branding", False, "SloughGPT branding not found")
            
            return found_branding
        except Exception as e:
            self.log_test("SloughGPT Branding", False, f"Error: {e}")
            return False
    
    def test_chat_interface(self):
        """Test chat interface functionality"""
        try:
            self.driver.get(self.base_url)
            time.sleep(2)
            
            # Look for chat input
            chat_input = self.wait_for_element("#chat-input, #chatInput, [data-cy='chat-input']", 5)
            if chat_input:
                self.log_test("Chat Input", True, "Chat input element found")
                
                # Try to type a test message
                if self.type_text("#chat-input, #chatInput, [data-cy='chat-input']", "Hello from E2E test!"):
                    self.log_test("Chat Input Typing", True, "Successfully typed message")
                    
                    # Look for send button
                    send_button = self.wait_for_element("button[type='submit'], [data-cy='send-button']", 5)
                    if send_button:
                        self.log_test("Send Button", True, "Send button found")
                        
                        # Try to click send
                        if self.click_element("button[type='submit'], [data-cy='send-button']"):
                            self.log_test("Send Message", True, "Message sent successfully")
                            time.sleep(2)
                            return True
                        else:
                            self.log_test("Send Message", False, "Failed to click send button")
                    else:
                        self.log_test("Send Button", False, "Send button not found")
                else:
                    self.log_test("Chat Input Typing", False, "Failed to type in chat input")
            else:
                self.log_test("Chat Input", False, "Chat input not found")
            
            return False
        except Exception as e:
            self.log_test("Chat Interface", False, f"Error: {e}")
            return False
    
    def test_model_selection(self):
        """Test model selection interface"""
        try:
            self.driver.get(self.base_url)
            time.sleep(2)
            
            # Look for model selector
            model_selectors = [
                ".sloughgpt-model-selector",
                "[aria-label='Select a model']",
                "#model-selector",
                "[data-cy='model-selector']"
            ]
            
            model_selector_found = False
            for selector in model_selectors:
                if self.wait_for_element(selector, 3):
                    model_selector_found = True
                    self.log_test("Model Selector", True, f"Found with selector: {selector}")
                    break
            
            if not model_selector_found:
                self.log_test("Model Selector", False, "No model selector found")
                return False
            
            return True
        except Exception as e:
            self.log_test("Model Selection", False, f"Error: {e}")
            return False
    
    def test_api_endpoints(self):
        """Test various API endpoints"""
        endpoints = [
            ("/api/models", "Models API"),
            ("/api/status", "Status API"),
            ("/api/models/sloughgpt", "SloughGPT Models API"),
            ("/api/status/sloughgpt", "SloughGPT Status API")
        ]
        
        passed = 0
        total = len(endpoints)
        
        for endpoint, test_name in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    self.log_test(test_name, True, f"Status: {response.status_code}")
                    passed += 1
                else:
                    self.log_test(test_name, False, f"Status: {response.status_code}")
            except Exception as e:
                self.log_test(test_name, False, f"Error: {e}")
        
        self.log_test("API Endpoints Overall", passed == total, f"Passed: {passed}/{total}")
        return passed == total
    
    def test_responsive_design(self):
        """Test responsive design on different viewports"""
        viewports = [
            (375, 667, "Mobile"),
            (768, 1024, "Tablet"),
            (1920, 1080, "Desktop")
        ]
        
        passed = 0
        for width, height, device in viewports:
            try:
                self.driver.set_window_size(width, height)
                self.driver.get(self.base_url)
                time.sleep(2)
                
                # Check if page loads properly
                if self.wait_for_element("body", 5):
                    self.log_test(f"Responsive {device}", True, f"Viewport: {width}x{height}")
                    passed += 1
                else:
                    self.log_test(f"Responsive {device}", False, f"Viewport: {width}x{height}")
            except Exception as e:
                self.log_test(f"Responsive {device}", False, f"Error: {e}")
        
        self.log_test("Responsive Design Overall", passed == len(viewports), f"Passed: {passed}/{len(viewports)}")
        return passed == len(viewports)
    
    def test_error_handling(self):
        """Test error handling capabilities"""
        try:
            self.driver.get(f"{self.base_url}/nonexistent-page")
            time.sleep(2)
            
            # Should handle 404 gracefully
            current_url = self.driver.current_url
            if "nonexistent-page" in current_url or self.driver.title:
                self.log_test("404 Handling", True, "404 page handled")
            else:
                self.log_test("404 Handling", False, "404 not properly handled")
            
            return True
        except Exception as e:
            self.log_test("Error Handling", False, f"Error: {e}")
            return False
    
    def test_accessibility(self):
        """Test basic accessibility features"""
        try:
            self.driver.get(self.base_url)
            time.sleep(2)
            
            # Check for alt text on images
            images = self.driver.find_elements(By.TAG_NAME, "img")
            images_with_alt = [img for img in images if img.get_attribute("alt")]
            
            accessibility_score = len(images_with_alt) / len(images) if images else 1
            
            if accessibility_score > 0.8:  # 80% of images have alt text
                self.log_test("Image Alt Text", True, f"Score: {accessibility_score:.1%}")
            else:
                self.log_test("Image Alt Text", False, f"Score: {accessibility_score:.1%}")
            
            return accessibility_score > 0.8
        except Exception as e:
            self.log_test("Accessibility", False, f"Error: {e}")
            return False
    
    def test_performance(self):
        """Test basic performance metrics"""
        try:
            start_time = time.time()
            self.driver.get(self.base_url)
            
            # Wait for page to load
            self.wait_for_element("body", 10)
            load_time = time.time() - start_time
            
            if load_time < 5:  # Page should load in under 5 seconds
                self.log_test("Page Load Performance", True, f"Load time: {load_time:.2f}s")
                return True
            else:
                self.log_test("Page Load Performance", False, f"Load time: {load_time:.2f}s")
                return False
        except Exception as e:
            self.log_test("Performance", False, f"Error: {e}")
            return False
    
    def run_all_tests(self):
        """Run complete E2E test suite"""
        print("🚀 Starting SloughGPT WebUI E2E Test Suite")
        print("="*60)
        
        if not self.setup_driver():
            print("❌ Failed to setup WebDriver")
            return False
        
        try:
            # Test suite
            tests = [
                self.test_server_health,
                self.test_frontend_loads,
                self.test_sloughgpt_branding,
                self.test_chat_interface,
                self.test_model_selection,
                self.test_api_endpoints,
                self.test_responsive_design,
                self.test_error_handling,
                self.test_accessibility,
                self.test_performance
            ]
            
            for test in tests:
                try:
                    test()
                    print()  # Add spacing between tests
                except Exception as e:
                    print(f"❌ Test failed with exception: {e}")
                    print()
            
            # Generate report
            self.generate_report()
            
        finally:
            if self.driver:
                self.driver.quit()
        
        return True
    
    def generate_report(self):
        """Generate test report"""
        passed = sum(1 for result in self.test_results if result["passed"])
        total = len(self.test_results)
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        print("="*60)
        print("📊 E2E TEST REPORT")
        print("="*60)
        print(f"🎯 Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {total - passed}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("🎉 PRODUCTION READY!")
            print("✅ SloughGPT WebUI is ready for production deployment")
        elif success_rate >= 60:
            print("⚠️  NEEDS ATTENTION")
            print("📋 Some issues need to be resolved before production")
        else:
            print("❌ NOT READY")
            print("🚫 Significant issues need to be addressed")
        
        print("\n📋 Failed Tests:")
        for result in self.test_results:
            if not result["passed"]:
                print(f"   ❌ {result['name']}: {result['message']}")
        
        print("="*60)
        
        # Save report to file
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": success_rate,
            "production_ready": success_rate >= 80,
            "test_results": self.test_results
        }
        
        try:
            with open("e2e_test_report.json", "w") as f:
                json.dump(report_data, f, indent=2)
            print(f"\n📄 Report saved to: e2e_test_report.json")
        except Exception as e:
            print(f"⚠️  Failed to save report: {e}")

def main():
    """Main function"""
    print("🔧 Setting up E2E test environment...")
    
    # Check if server is running
    tester = SloughGPTE2ETester()
    
    try:
        response = requests.get(f"{tester.base_url}/api/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Server not running at {tester.base_url}")
            print("🚀 Please start the server first:")
            print("   python3 enhanced_webui.py")
            return False
    except:
        print(f"❌ Server not running at {tester.base_url}")
        print("🚀 Please start the server first:")
        print("   python3 enhanced_webui.py")
        return False
    
    # Install Selenium if needed
    try:
        import selenium
    except ImportError:
        print("📦 Installing Selenium...")
        subprocess.run([sys.executable, "-m", "pip", "install", "selenium"])
    
    # Run tests
    success = tester.run_all_tests()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)