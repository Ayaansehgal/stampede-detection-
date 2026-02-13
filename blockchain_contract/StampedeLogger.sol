// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StampedeLogger {

    struct EventRecord {
        uint256 timestamp;
        string location;
        string label;
        uint256 riskScore;
        string dataHash;
    }

    EventRecord[] public records;

    function logEvent(
        uint256 _timestamp,
        string memory _location,
        string memory _label,
        uint256 _riskScore,
        string memory _dataHash
    ) public {

        records.push(EventRecord(
            _timestamp,
            _location,
            _label,
            _riskScore,
            _dataHash
        ));
    }

    function getTotalEvents() public view returns (uint256) {
        return records.length;
    }
}