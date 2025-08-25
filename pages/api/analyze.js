export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).send('Method not allowed');

  const { resume, jobDescription } = req.body;

  // Dummy logic: just returns a fake analysis
  const results = {
    matchScore: 67,
    missingKeywords: ['Agile', 'Docker', 'CI/CD'],
    suggestedRephrases: [
      { original: 'Worked on cloud infra', suggestion: 'Implemented scalable AWS cloud infrastructure' }
    ]
  };

  res.status(200).json(results);
}
