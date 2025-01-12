# examples/theme_discovery_example.py
from typing import List
import json
from models.article import Article
from main import ThemeDiscoveryOrchestrator


def load_sample_articles() -> List[Article]:
    """Load sample articles from JSON file"""
    with open('sample_articles.json', 'r') as f:
        data = json.load(f)

    articles = []
    for item in data:
        articles.append(Article(
            id=item['id'],
            path=item['path'],
            keywords=[(kw[0], kw[1]) for kw in item['keywords']]
        ))

    return articles


def main():
    # Load sample articles
    articles = load_sample_articles()

    # Initialize orchestrator
    orchestrator = ThemeDiscoveryOrchestrator(
        n_themes=None,  # Auto-determine number of themes
        min_coherence=0.1,
        assignment_threshold=0.2
    )

    # Discover themes
    themes = orchestrator.discover_themes(articles)

    # Generate reports
    print("\nText Report:")
    print(orchestrator.generate_report(include_visualizations=True))

    print("\nGenerating HTML report...")
    html_report = orchestrator.generate_report(output_format='html')
    with open('theme_analysis_report.html', 'w') as f:
        f.write(html_report)

    print("Analysis complete! Check theme_analysis_report.html for detailed results.")


if __name__ == "__main__":
    main()