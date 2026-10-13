import unittest

from multi_agents import BillingAgent, GeneralAgent, TechAgent


class KeywordFallbackTests(unittest.TestCase):
    def test_generic_technical_query_returns_multiple_categories(self):
        agent = TechAgent()
        result = agent._match_tech_info("设备无法开机")
        for category in agent.tech_database:
            self.assertIn(category, result)

    def test_generic_billing_query_returns_multiple_categories(self):
        agent = BillingAgent()
        result = agent._match_billing_info("退款账单咨询")
        for category in agent.billing_database:
            self.assertIn(category, result)

    def test_generic_contact_query_returns_multiple_categories(self):
        agent = GeneralAgent()
        result = agent._match_service_info("客服电话是多少")
        for category in agent.service_database:
            self.assertIn(category, result)


if __name__ == "__main__":
    unittest.main()
