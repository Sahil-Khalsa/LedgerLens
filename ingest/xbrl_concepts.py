"""
Alias map: canonical short name → list of XBRL concept names companies actually use.
Different filers report the same economic line item under different concept names.
The verifier uses this map to find the right concept before comparing values.
"""

CONCEPT_ALIASES: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenueNet",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "NetIncomeLossAvailableToCommonStockholdersDiluted",
    ],
    "gross_profit": [
        "GrossProfit",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "operating_expenses": [
        "OperatingExpenses",
        "CostsAndExpenses",
    ],
    "cost_of_revenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],
    "research_and_development": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
    "selling_general_admin": [
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
    ],
    "eps_basic": [
        "EarningsPerShareBasic",
    ],
    "eps_diluted": [
        "EarningsPerShareDiluted",
    ],
    "shares_basic": [
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ],
    "shares_diluted": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ],
    "total_assets": [
        "Assets",
    ],
    "total_liabilities": [
        "Liabilities",
    ],
    "total_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalentsAndShortTermInvestments",
    ],
    "long_term_debt": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermNotesPayable",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpendituresIncurredButNotYetPaid",
        "PaymentsForCapitalImprovements",
    ],
    "depreciation_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
    "free_cash_flow": [
        "FreeCashFlow",  # non-standard; most filers don't tag this
    ],
    "inventory": [
        "InventoryNet",
        "InventoryGross",
    ],
    "accounts_receivable": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
    ],
}

# Reverse map: concept name → canonical key (built once at import)
CONCEPT_TO_CANONICAL: dict[str, str] = {
    concept: canonical for canonical, concepts in CONCEPT_ALIASES.items() for concept in concepts
}


def resolve_canonical(concept_name: str) -> str | None:
    """Return the canonical alias key for a raw XBRL concept name, or None."""
    return CONCEPT_TO_CANONICAL.get(concept_name)
