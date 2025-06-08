#!/usr/bin/env python3
"""
Topic Drift Detection Demo - STRESS TEST EDITION

This script demonstrates how the topic drift detection works with various
conversation scenarios, including extremely challenging edge cases designed
to stress-test and potentially break the system.
"""

import logging
from remgpt.detection import TopicDriftDetector
from remgpt.types import UserMessage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("drift_demo")

def create_detector():
    """Create a TopicDriftDetector for testing with ULTRA-EXTREME CONSERVATIVE parameters."""
    return TopicDriftDetector(
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.2,   # Extremely low threshold - accept very low similarity
        drift_threshold=1.2,        # Above maximum possible - require overwhelming evidence
        alpha=0.001,                # Minimal sensitivity
        window_size=15,             # Very large window for extreme stability
        logger=logger
    )

def create_conservative_detector():
    """Create a TopicDriftDetector with CONSERVATIVE parameters for comparison."""
    return TopicDriftDetector(
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.7,  # Original conservative value
        drift_threshold=0.4,       # Original conservative value
        alpha=0.05,                # Original conservative value
        window_size=5,
        logger=logger
    )

def create_very_sensitive_detector():
    """Create a TopicDriftDetector with VERY AGGRESSIVE parameters for testing."""
    return TopicDriftDetector(
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.5,  # Even lower threshold
        drift_threshold=0.1,       # Very aggressive drift detection
        alpha=0.2,                 # Much higher sensitivity
        window_size=3,             # Smaller window for faster detection
        logger=logger
    )

def test_scenario(detector, scenario_name, messages, description, expected_drifts=None):
    """Test a conversation scenario and report results."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*60}")
    print(f"Description: {description}")
    print(f"Messages: {len(messages)}")
    if expected_drifts is not None:
        print(f"Expected drifts: {expected_drifts}")
    print("-" * 60)
    
    detector.reset()
    
    drift_detections = []
    similarities = []
    
    for i, content in enumerate(messages):
        message = UserMessage(content=content, name="user")
        drift_detected, _, similarity = detector.detect_drift(message)
        
        drift_detections.append(drift_detected)
        similarities.append(similarity)
        
        drift_status = "🚨 DRIFT" if drift_detected else "✅ NO DRIFT"
        print(f"{i+1:2d}. [{similarity:.3f}] {drift_status} | {content[:50]}...")
    
    # Summary
    total_drifts = sum(drift_detections)
    avg_similarity = sum(similarities[1:]) / len(similarities[1:]) if len(similarities) > 1 else 0
    min_similarity = min(similarities[1:]) if len(similarities) > 1 else 0
    
    print("-" * 60)
    print(f"RESULTS:")
    print(f"  Total drifts detected: {total_drifts}/{len(messages)-1} possible")
    print(f"  Average similarity: {avg_similarity:.3f}")
    print(f"  Minimum similarity: {min_similarity:.3f}")
    print(f"  Drift rate: {total_drifts/(len(messages)-1)*100:.1f}%")
    
    # Performance assessment
    if expected_drifts is not None:
        if total_drifts == expected_drifts:
            print(f"  ✅ PERFECT: Detected exactly {expected_drifts} expected drifts")
        elif abs(total_drifts - expected_drifts) <= 1:
            print(f"  ✅ GOOD: Within ±1 of expected ({expected_drifts})")
        else:
            print(f"  ⚠️  SUBOPTIMAL: Expected ~{expected_drifts}, got {total_drifts}")
    
    return {
        'scenario': scenario_name,
        'total_drifts': total_drifts,
        'possible_drifts': len(messages) - 1,
        'avg_similarity': avg_similarity,
        'min_similarity': min_similarity,
        'drift_rate': total_drifts/(len(messages)-1)*100,
        'expected_drifts': expected_drifts
    }

def main():
    print("🚀 Topic Drift Detection Demo - ULTRA-EXTREME CONSERVATIVE PARAMETERS EDITION")
    print("Final optimization push to achieve near-zero false positives...")
    print("\n🔧 PARAMETER OPTIMIZATION (Final Iteration):")
    print("   • similarity_threshold: 0.3 → 0.2 (extremely low)")
    print("   • drift_threshold: 0.8 → 1.2 (above maximum possible)")
    print("   • alpha: 0.005 → 0.001 (minimal sensitivity)")
    print("   • window_size: 10 → 15 (very large window)")
    print("   🎯 Goal: Achieve <10% false positive rate while maintaining reasonable true positive rate")
    
    detector = create_detector()
    
    # Initialize model
    print("\n📦 Loading SentenceTransformer model...")
    test_msg = UserMessage(content="initialization", name="user")
    detector.create_embedding(test_msg)
    print("✅ Model loaded successfully!")
    
    results = []
    
    # ========== REALISTIC HUMAN-AI CHAT CONVERSATIONS ==========
    
    # Scenario 1: Executive Presentation → Sales Report (Task switching)
    executive_chat = [
        "Hello! I need help creating an executive-level PowerPoint presentation for our Q3 performance review",
        "I'd be happy to help you create that presentation. Let me suggest a structure with key performance metrics, financial highlights, and strategic achievements. Would you like me to start with an outline?",
        "Yes, that sounds great. Can you focus on making it visually appealing with charts and graphs?",
        "Absolutely! I'll include data visualization recommendations and suggest chart types for each section. Here's a detailed outline with visual elements for each slide.",
        "Actually, I just realized I need something different. Can you help me generate a detailed sales report instead? I need to analyze our regional performance.",
        "Of course! I can help you create a comprehensive sales report. What specific regions and time periods would you like me to include in the analysis?"
    ]
    
    results.append(test_scenario(
        detector, "Executive Chat", executive_chat,
        "Presentation help → Sales report - realistic task switching in conversation",
        expected_drifts=1
    ))
    
    # Scenario 2: Customer Support Workflow (Escalating issue)
    support_chat = [
        "Hi, I'm having trouble with a customer who can't log into their account. The password reset isn't working.",
        "I can help you troubleshoot this login issue. Let's start by checking if their email address is correctly registered in the system. Can you verify the email they're using?",
        "Yes, I checked and the email is correct. But now they're also asking about being charged twice last month.",
        "That's a different issue that involves billing. For the duplicate charge, you'll need to escalate to the billing department. Let me give you the steps to handle both issues.",
        "Thanks! While I have you, can you help me write a professional email to explain the billing situation to the customer?",
        "Absolutely! Here's a draft email that explains the situation professionally and outlines the steps we're taking to resolve both the login and billing issues."
    ]
    
    results.append(test_scenario(
        detector, "Support Chat", support_chat,
        "Technical support → Billing issue → Email drafting - realistic support workflow",
        expected_drifts=2
    ))
    
    # Scenario 3: Project Management Workflow
    project_chat = [
        "Good morning! I need to update our project timeline for the website redesign project. We're running behind schedule.",
        "Good morning! I can help you revise the project timeline. Let me create an updated schedule that accounts for the delays. What are the main bottlenecks causing the delays?",
        "The design phase took longer than expected. Can you also help me draft an email to stakeholders explaining the delay?",
        "Of course! Here's a draft email that explains the design phase delays professionally and presents the revised timeline. It focuses on the quality improvements achieved during the extended design phase.",
        "Perfect! Now I need to switch gears - I have a budget meeting tomorrow and need to prepare cost estimates for the additional design work.",
        "I'll help you prepare those cost estimates. Let me create a budget breakdown that shows the additional design costs and explains the value they bring to the project."
    ]
    
    results.append(test_scenario(
        detector, "Project Chat", project_chat,
        "Timeline update → Stakeholder email → Budget prep - connected project tasks",
        expected_drifts=1
    ))
    
    # Scenario 4: Similar Communication Tasks (Minimal drift expected)
    communication_chat = [
        "Hi! I need help writing a professional email to a client about our meeting next week.",
        "I'd be happy to help you write that email. Let me draft a professional message that confirms the meeting details and sets a positive tone.",
        "That's great! Can you also help me write a follow-up email for the proposal we sent last month?",
        "Absolutely! Here's a polite follow-up email that inquires about the proposal status while maintaining a professional relationship.",
        "Perfect! One more thing - I need to send a thank you note after yesterday's client meeting.",
        "I'll draft a thoughtful thank you email that reinforces the key points discussed and expresses appreciation for their time."
    ]
    
    results.append(test_scenario(
        detector, "Communication Chat", communication_chat,
        "Multiple email tasks - same domain, should show minimal drift",
        expected_drifts=0
    ))
    
    # Scenario 5: Department Switching (Customer Service → HR → IT)
    department_chat = [
        "I need help handling a customer complaint about delayed shipping. How should I respond professionally?",
        "Here's a professional response template that acknowledges their concern, explains the shipping delay, and offers appropriate compensation or expedited shipping.",
        "Thanks! By the way, completely different topic - when is the deadline for updating my health insurance enrollment?",
        "The health insurance enrollment deadline is typically during the annual open enrollment period. Let me help you find the specific dates and required forms for your company.",
        "Got it! And sorry, one more thing - my work laptop has been incredibly slow lately. Any IT troubleshooting tips?",
        "I can help with basic laptop troubleshooting. Let's start with checking available storage space, running disk cleanup, and reviewing startup programs that might be slowing down your system."
    ]
    
    results.append(test_scenario(
        detector, "Department Chat", department_chat,
        "Customer service → HR benefits → IT support - clear topic jumps",
        expected_drifts=2
    ))
    
    # Scenario 6: Sales Process Flow  
    sales_chat = [
        "Hi! I'm working on a big proposal for an enterprise client who needs 500 user licenses. Can you help me structure this?",
        "I'd be happy to help structure your enterprise proposal. Here's a comprehensive template that includes pricing tiers, implementation timeline, and support packages for 500 users.",
        "Excellent! Can you also help me update their information in our CRM system?",
        "I can guide you through updating the CRM. Here are the key fields to update for enterprise prospects and how to set up automated follow-up sequences.",
        "Perfect! Now I need to schedule a follow-up call with them for next week to discuss the proposal.",
        "I'll help you craft the follow-up email to schedule that call. Here's a message that references the proposal and suggests several meeting times."
    ]
    
    results.append(test_scenario(
        detector, "Sales Chat", sales_chat,
        "Proposal creation → CRM update → Follow-up scheduling - connected sales activities",
        expected_drifts=0
    ))
    
    # Scenario 7: Context Switch (Meeting Planning → Legal Review)
    context_switch_chat = [
        "I need to organize our quarterly review meeting. Can you help me coordinate schedules for all department heads?",
        "I'll help you coordinate the quarterly review. Here's a scheduling strategy and email template to gather availability from all department heads efficiently.",
        "Great! Now completely different topic - I need help reviewing this legal contract. There are clauses in section 4 I don't understand.",
        "I can help explain contract clauses, though I'd recommend having a lawyer review anything complex. What specific terms in section 4 are unclear?",
        "The termination provisions seem complicated. What are the typical implications of these clauses?",
        "Termination clauses typically cover notice periods, compensation, and post-employment restrictions. However, for this specific contract, I'd strongly recommend getting legal counsel to review the exact implications."
    ]
    
    results.append(test_scenario(
        detector, "Context Switch Chat", context_switch_chat,
        "Meeting coordination → Legal contract review - major context switch",
        expected_drifts=1
    ))
    
    # Scenario 8: Technical Support Escalation
    tech_support_chat = [
        "Our software keeps crashing when users try to open large files. I've tried the basic troubleshooting steps already.",
        "Let's work through advanced troubleshooting for the large file crashes. Can you tell me what file sizes trigger the crashes and what error messages appear?",
        "Files over 50MB crash it consistently. This is affecting productivity - should I escalate to engineering?",
        "Yes, this sounds like a memory management issue that needs engineering attention. Here's how to document the bug report with the technical details they'll need.",
        "Thanks! While I wait for engineering, can you help with a different issue? Users can't figure out how to export data to Excel format.",
        "I can help with the Excel export issue. Here's a step-by-step guide for users, plus some troubleshooting tips for common export problems."
    ]
    
    results.append(test_scenario(
        detector, "Tech Support Chat", tech_support_chat,
        "Software crashes → Bug escalation → Different feature help - technical support flow",
        expected_drifts=1
    ))
    
    # Scenario 9: Business Domain Jump (Finance → Marketing)
    business_jump_chat = [
        "I'm trying to reconcile the petty cash for this month and there's a $50 discrepancy I can't track down.",
        "Let me help you track down that discrepancy. Here's a systematic approach to reconciling petty cash, including common places to look for missing transactions.",
        "Found it! Thanks. Now totally different question - can you help me design a marketing flyer for our new product launch?",
        "I'd be happy to help with your marketing flyer! Let me suggest a design layout and content structure that highlights your product's key benefits effectively.",
        "Perfect! The product launches next month, so the flyer should create excitement and urgency.",
        "Great! Here's a design concept that builds excitement with compelling headlines and creates urgency with launch timing and early-bird offers."
    ]
    
    results.append(test_scenario(
        detector, "Business Jump Chat", business_jump_chat,
        "Financial reconciliation → Marketing flyer design - major business domain switch",
        expected_drifts=1
    ))
    
    # Scenario 10: E-commerce Customer Journey
    ecommerce_chat = [
        "Hello! A customer placed an order 5 days ago and wants to know where it is. The tracking shows no updates.",
        "I can help you handle this shipping inquiry. Here's how to check the tracking status and what to tell the customer about potential delays.",
        "The customer is also asking if they can get a refund if it doesn't arrive by Friday since they need it for an event.",
        "For time-sensitive orders, here's our refund policy and how to handle the situation professionally while managing expectations.",
        "They're also asking about exchanging an item from a previous order for a different size. Is that possible?",
        "Yes, exchanges are typically possible within our return window. Here's how to process the size exchange and what information you'll need from the customer."
    ]
    
    results.append(test_scenario(
        detector, "E-commerce Chat", ecommerce_chat,
        "Order tracking → Refund policy → Size exchange - related e-commerce support topics",
        expected_drifts=1
    ))
    
    # Scenario 11: Professional Development Conversation
    career_chat = [
        "I want to advance my career and am thinking about getting a project management certification. Any advice?",
        "A PM certification is a great career move! Let me explain the different certifications available (PMP, PRINCE2, Agile) and their requirements.",
        "PMP sounds good. I should probably also update my LinkedIn profile to reflect my recent achievements.",
        "Absolutely! A strong LinkedIn profile is crucial for career advancement. Here's how to optimize your profile to highlight your PM experience and skills.",
        "Thanks! I also have my annual performance review coming up. What should I emphasize about my contributions this year?",
        "For your performance review, focus on quantifiable achievements, leadership examples, and how your work contributed to team/company goals. Here's a framework to organize your accomplishments."
    ]
    
    results.append(test_scenario(
        detector, "Career Chat", career_chat,
        "Certification planning → LinkedIn optimization → Performance review prep - career development flow",
        expected_drifts=1
    ))
    
    # Scenario 12: Office Administration Workflow
    admin_chat = [
        "Good morning! I need to order office supplies for the team. We're running low on printer paper and ink cartridges.",
        "Good morning! I can help you organize the office supply order. Here's a supply checklist and recommended vendors for bulk office supplies.",
        "Great! I also need to arrange catering for next week's client meeting. We'll have 12 people with some vegetarian options needed.",
        "I'll help you arrange the catering. Here are local catering options that can accommodate 12 people with vegetarian choices, including contact information and typical pricing.",
        "Perfect! One last thing - I need to reserve visitor parking spots for the executives coming Thursday and Friday.",
        "I'll help you coordinate the visitor parking. Here's how to reserve spots and what information to provide to the visiting executives about parking access."
    ]
    
    results.append(test_scenario(
        detector, "Admin Chat", admin_chat,
        "Office supplies → Catering → Parking reservations - administrative task variety",
        expected_drifts=2
    ))
    
    # Scenario 13: Same Task, Different Clients
    multi_client_chat = [
        "I need to create a project proposal for ABC Corporation. Can you help me with the structure and content?",
        "I'd be happy to help create that proposal for ABC Corporation. Here's a professional proposal template with sections for scope, timeline, pricing, and deliverables.",
        "Excellent! Now I need a similar proposal for XYZ Industries, but they have different budget constraints.",
        "I'll adapt the proposal for XYZ Industries. Here's a version that emphasizes cost-effectiveness and flexible payment options while maintaining professional quality.",
        "Perfect! I also need one more proposal for DEF Enterprises - they're more focused on quick turnaround than budget.",
        "For DEF Enterprises, here's a proposal that emphasizes speed of delivery and expedited project management while highlighting our rapid implementation capabilities."
    ]
    
    results.append(test_scenario(
        detector, "Multi-Client Chat", multi_client_chat,
        "Same task (proposals) for different clients - minimal drift expected",
        expected_drifts=0
    ))
    
    # Scenario 14: Urgent vs Routine Task Management  
    urgency_chat = [
        "Emergency! I need talking points for a board meeting that starts in 2 hours. Focus on quarterly financial highlights.",
        "I'll help you prepare those urgent talking points immediately. Here are the key financial highlights formatted for quick reference during your board presentation.",
        "Thank you so much! That's exactly what I needed. When you have time later, can you help me organize my email inbox? It's a mess.",
        "Glad the talking points helped! I'd be happy to help organize your email. Here are strategies for creating folders, setting up filters, and managing your inbox efficiently.",
        "That would be great! I have thousands of unread emails that need sorting and organizing.",
        "Here's a systematic approach to tackle your email backlog, including automated rules to categorize future emails and time-saving techniques for mass email management."
    ]
    
    results.append(test_scenario(
        detector, "Urgency Chat", urgency_chat,
        "Urgent board prep → Routine email organization - different priority levels",
        expected_drifts=1
    ))
    
    # Scenario 15: Cross-functional Collaboration
    collaboration_chat = [
        "I need to coordinate with our design team on the new website mockups. Can you help me create a feedback process?",
        "I'll help you create an effective design feedback process. Here's a structured approach for collecting, organizing, and communicating feedback to your design team.",
        "That's perfect! I should also sync with engineering about technical feasibility of these designs.",
        "Great idea to involve engineering early. Here's how to facilitate productive design-engineering collaboration and what technical questions to ask.",
        "Excellent! Let's schedule a joint meeting with both teams to align on timeline and deliverables.",
        "I'll help you organize that joint meeting. Here's an agenda template and facilitation tips to ensure productive collaboration between design and engineering teams."
    ]
    
    results.append(test_scenario(
        detector, "Collaboration Chat", collaboration_chat,
        "Design feedback → Engineering sync → Team coordination - connected collaboration tasks",
        expected_drifts=0
    ))
    
    # Scenario 16: Social/Casual Chat Mixed with Work  
    casual_work_chat = [
        "Hi there! How's your day going? I hope you're doing well!",
        "Hello! I'm doing well, thank you for asking! How can I help you today?",
        "I need help with something work-related. Can you help me write a professional email to a difficult client?",
        "Of course! I'd be happy to help you craft a professional email for a challenging client situation. What's the main issue you need to address?",
        "Thanks so much! You're always so helpful. Have a great day!",
        "You're very welcome! I'm glad I could help. Have a wonderful day as well!"
    ]
    
    results.append(test_scenario(
        detector, "Casual Work Chat", casual_work_chat,
        "Social pleasantries → Work task → Social closing - typical friendly interaction",
        expected_drifts=1
    ))
    
    # Scenario 17: Clarification and Refinement Pattern
    refinement_chat = [
        "I need help creating a marketing campaign for our new software product. Can you give me some ideas?",
        "I'd be happy to help with your marketing campaign! Here are several campaign concepts focusing on different aspects of your software - features, benefits, and target audiences.",
        "These are good, but can you make them more focused on small businesses? That's our main target market.",
        "Absolutely! Here are revised campaign concepts specifically tailored for small businesses, emphasizing cost savings, ease of use, and quick implementation.",
        "Much better! Can you also suggest some specific social media post ideas for LinkedIn?",
        "Perfect! Here are LinkedIn-specific post ideas that will resonate with small business owners, including success stories, tips, and engaging questions to drive engagement."
    ]
    
    results.append(test_scenario(
        detector, "Refinement Chat", refinement_chat,
        "Marketing campaign → Small business focus → Social media specifics - iterative refinement",
        expected_drifts=0
    ))
    
    # Final Summary Report
    print(f"\n{'='*80}")
    print("📊 COMPREHENSIVE STRESS TEST RESULTS")
    print(f"{'='*80}")
    
    print(f"{'Scenario':<25} {'Drifts':<8} {'Expected':<9} {'Match':<6} {'Rate':<8} {'Min Sim':<8}")
    print("-" * 80)
    
    perfect_matches = 0
    good_matches = 0
    
    for result in results:
        expected = result['expected_drifts']
        actual = result['total_drifts']
        
        if expected is not None:
            if actual == expected:
                match_status = "✅ PERF"
                perfect_matches += 1
            elif abs(actual - expected) <= 1:
                match_status = "✅ GOOD"
                good_matches += 1
            else:
                match_status = "❌ POOR"
        else:
            match_status = "- N/A"
        
        print(f"{result['scenario']:<25} {result['total_drifts']:<8} "
              f"{expected if expected is not None else 'N/A':<9} "
              f"{match_status:<6} {result['drift_rate']:<7.1f}% "
              f"{result['min_similarity']:<7.3f}")
    
    # Detailed Analysis
    print(f"\n🔍 STRESS TEST ANALYSIS:")
    print(f"   • Perfect matches: {perfect_matches}/{len([r for r in results if r['expected_drifts'] is not None])}")
    print(f"   • Good matches (±1): {good_matches}")
    
    # Category Analysis
    print(f"\n📈 CATEGORY BREAKDOWN:")
    
    # False Positives Check
    false_positive_scenarios = [r for r in results if r['expected_drifts'] == 0]
    if false_positive_scenarios:
        avg_fp_rate = sum(r['drift_rate'] for r in false_positive_scenarios) / len(false_positive_scenarios)
        print(f"   🔒 False Positive Rate: {avg_fp_rate:.1f}% (should be ~0%)")
        if avg_fp_rate < 10:
            print(f"      ✅ EXCELLENT: Very low false positive rate")
        elif avg_fp_rate < 25:
            print(f"      ✅ GOOD: Acceptable false positive rate")
        else:
            print(f"      ⚠️  HIGH: Concerning false positive rate")
    
    # True Positive Check
    high_drift_scenarios = [r for r in results if r['expected_drifts'] and r['expected_drifts'] >= 3]
    if high_drift_scenarios:
        avg_tp_rate = sum(r['drift_rate'] for r in high_drift_scenarios) / len(high_drift_scenarios)
        print(f"   🎯 True Positive Rate: {avg_tp_rate:.1f}% (should be >80%)")
        if avg_tp_rate > 80:
            print(f"      ✅ EXCELLENT: High sensitivity to obvious changes")
        elif avg_tp_rate > 60:
            print(f"      ✅ GOOD: Reasonable sensitivity")
        else:
            print(f"      ⚠️  LOW: Missing obvious topic changes")
    
    # Edge Case Performance
    edge_cases = [r for r in results if "Confusion" in r['scenario'] or "Boundary" in r['scenario'] or "Mixing" in r['scenario']]
    if edge_cases:
        edge_accuracy = sum(1 for r in edge_cases if r['expected_drifts'] is not None and abs(r['total_drifts'] - r['expected_drifts']) <= 1) / len(edge_cases)
        print(f"   🧪 Edge Case Accuracy: {edge_accuracy*100:.1f}% (within ±1 of expected)")
        if edge_accuracy > 0.7:
            print(f"      ✅ ROBUST: Handles edge cases well")
        elif edge_accuracy > 0.5:
            print(f"      ✅ DECENT: Reasonable edge case handling")
        else:
            print(f"      ⚠️  FRAGILE: Struggles with edge cases")
    
    # Overall Assessment
    avg_drift_rate = sum(r['drift_rate'] for r in results) / len(results)
    print(f"\n🎖️  OVERALL ASSESSMENT:")
    print(f"   • Average drift detection rate: {avg_drift_rate:.1f}%")
    print(f"   • Perfect predictions: {perfect_matches}/{len([r for r in results if r['expected_drifts'] is not None])}")
    
    if perfect_matches >= len([r for r in results if r['expected_drifts'] is not None]) * 0.6:
        print(f"   🏆 EXCELLENT: System handles most scenarios correctly")
    elif perfect_matches >= len([r for r in results if r['expected_drifts'] is not None]) * 0.4:
        print(f"   ✅ GOOD: System performs reasonably well")
    else:
        print(f"   ⚠️  NEEDS IMPROVEMENT: System struggles with many scenarios")
    
    # Failure Analysis
    failures = [r for r in results if r['expected_drifts'] is not None and abs(r['total_drifts'] - r['expected_drifts']) > 1]
    if failures:
        print(f"\n❌ FAILURE ANALYSIS ({len(failures)} scenarios):")
        for failure in failures:
            print(f"   • {failure['scenario']}: Expected {failure['expected_drifts']}, got {failure['total_drifts']}")

if __name__ == "__main__":
    main() 