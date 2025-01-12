# main.py
import json
from utils.json_encoder import ThemeEncoder

from typing import List, Dict
from models.article import Article
from models.theme import Theme
from agents.text_processor_agent import TextProcessorAgent
from agents.feature_extractor_agent import FeatureExtractorAgent
from agents.theme_discovery_agent import ThemeDiscoveryAgent
from agents.coherence_agent import CoherenceAgent
from agents.assignment_agent import AssignmentAgent
from agents.visualization_agent import VisualizationAgent


class ThemeDiscoveryOrchestrator:
    def __init__(self,
                 n_themes: int = None,
                 min_coherence: float = 0.1,
                 assignment_threshold: float = 0.2):
        self.text_processor = TextProcessorAgent()
        self.feature_extractor = FeatureExtractorAgent()
        self.theme_discoverer = ThemeDiscoveryAgent(n_themes=n_themes)
        self.coherence_agent = CoherenceAgent()
        self.assignment_agent = AssignmentAgent(threshold=assignment_threshold)
        self.visualization_agent = VisualizationAgent()

        self.min_coherence = min_coherence
        self.themes = None

    def discover_themes(self, articles: List[Article]) -> List[Theme]:
        """Main method to discover and analyze themes"""
        # Process text
        for article in articles:
            article.title = self.text_processor.process_title(article.filename)

        # Extract features
        title_features = self.feature_extractor.extract_title_features(articles)
        keyword_features = self.feature_extractor.extract_keyword_features(articles)
        combined_features = self.feature_extractor.combine_features(
            title_features, keyword_features
        )

        # Discover themes
        W, H = self.theme_discoverer.discover_themes(combined_features)

        # Assign articles to themes
        assignments = self.assignment_agent.assign_articles_to_themes(W, articles)

        # Create themes
        self.themes = self.theme_discoverer.create_themes(
            combined_features, assignments
        )

        # Calculate coherence
        for theme in self.themes:
            theme.coherence = self.coherence_agent.calculate_theme_coherence(theme)

        # Filter low coherence themes
        self.themes = [theme for theme in self.themes
                       if theme.coherence >= self.min_coherence]

        return self.themes

    def generate_report(self,
                        include_visualizations: bool = True,
                        output_format: str = 'text') -> str:
        """Generate analysis report"""
        if self.themes is None:
            return "No themes discovered yet. Run discover_themes() first."

        if output_format == 'html':
            return self.visualization_agent.generate_html_report(self.themes)

        if include_visualizations:
            self.visualization_agent.plot_theme_distribution(self.themes)
            self.visualization_agent.plot_theme_network(self.themes)

        # Text report
        report = "\nTheme Analysis Report\n" + "=" * 20 + "\n"

        for theme in self.themes:
            report += f"\nTheme {theme.id + 1}:\n"
            report += f"Key terms: {', '.join(theme.top_terms)}\n"
            report += f"Coherence: {theme.coherence:.3f}\n"
            report += f"Number of articles: {len(theme.articles)}\n"

            if theme.articles:
                report += "The articles:\n"
                for article in sorted(theme.articles,
                                      key=lambda x: x['score'],
                                      reverse=True):
                    if article['score'] < 0.01:
                        break
                    report += f"  - {article['path']} (score: {article['score']:.2f})\n"

        with open('theme_analysis_report.json', 'w') as f:
            f.write(json.dumps(self.themes, cls=ThemeEncoder))

        return report